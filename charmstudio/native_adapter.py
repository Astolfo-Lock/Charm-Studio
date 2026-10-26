"""Compile an independent, skinned charm for the native viewmodel renderer.

Requires Left 4 Dead 2 Authoring Tools (Steam app 563), including studiomdl.exe.
The compiler runs against an isolated game directory under workspace/.build;
the installed game and the weapon's resources are never written or packaged.
Placement transforms are applied by the native runtime, not baked into vertices.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import numpy as np

from .bake import _materials
from .charm_pose import reference_pose, reference_vertices, skin_directions
from .project import digest
from .source import Model
from .vpk import VPK, write_vpk


# L4D2's struct is flags + 29 floats (120 bytes), without Source 2013's
# additional five boing parameters. Source: alliedmodders/hl2sdk branch l4d2,
# public/studio.h, mstudiojigglebone_t. Do not use the 140-byte SDK2013 layout.
L4D2_JIGGLE = struct.Struct('<i29f')


@dataclass(frozen=True)
class NativeAdapterExport:
    model_path: str
    vpk_path: Path
    work_dir: Path
    files: tuple[str, ...]


def _quote(value):
    value = str(value).replace('\\', '/')
    if any(char in value for char in ('"', '\n', '\r', '\0')):
        raise ValueError('Identificador QC/SMD no válido')
    return '"' + value + '"'


def _numbers(values):
    return ' '.join(f'{float(value):.9f}' for value in values)


def _bone_euler(matrix):
    """Source bone XYZ radians, Rz(z) Ry(y) Rx(x), including gimbal lock."""
    horizontal = np.hypot(matrix[0, 0], matrix[1, 0])
    y = np.arctan2(-matrix[2, 0], horizontal)
    if horizontal > 1e-8:
        return [np.arctan2(matrix[2, 1], matrix[2, 2]), y,
                np.arctan2(matrix[1, 0], matrix[0, 0])]
    return [0, y, np.arctan2(-matrix[0, 1], matrix[1, 1])]


def _jiggle_payloads(model):
    count, offset = model.r.unpack('ii', 156)
    result = {}
    for index in range(count):
        bone = offset + index*216
        kind, relative = model.r.unpack('ii', bone+164)
        if not kind:
            continue
        if kind != 5:
            raise ValueError(f'Hueso procedural no admitido: {model.bones[index].name} (tipo {kind})')
        start = bone+relative
        if relative == 0 or start < 0 or start+L4D2_JIGGLE.size > len(model.data):
            raise ValueError('Parámetros jiggle truncados')
        result[model.bones[index].name] = model.data[start:start+L4D2_JIGGLE.size]
    return result


def _restore_jiggles(data, payloads):
    if not payloads:
        return data
    compiled = Model(lambda path: data, 'models/compiled.mdl', geometry=False)
    offset = compiled.r.i(160)
    mutable = bytearray(data)
    names = {bone.name: index for index, bone in enumerate(compiled.bones)}
    for name, payload in payloads.items():
        if name not in names:
            raise ValueError(f'El compilador eliminó el hueso jiggle {name}')
        bone = offset + names[name]*216
        mutable.extend(b'\0'*(-len(mutable) % 4))
        start = len(mutable)
        mutable.extend(payload)
        struct.pack_into('<i', mutable, bone+160, compiled.r.i(bone+160) | 4)
        struct.pack_into('<ii', mutable, bone+164, 5, start-bone)
    struct.pack_into('<i', mutable, 76, len(mutable))
    return bytes(mutable)


def _namespace_materials(names, assets, identity):
    """Isolate repaired VMTs and their include graph from subscribed addons."""
    prefix = f'charms_studio/native/{identity}/'
    mapping = {}
    for index, name in enumerate(names):
        original = 'materials/' + name.replace('\\', '/').lower() + '.vmt'
        mapping.setdefault(original, prefix + f'm{index}')
    for path in sorted(assets):
        if path.endswith('.vmt') and path not in mapping:
            mapping[path] = prefix + f'd{len(mapping)}'
    include = re.compile(r'(?im)//[^\r\n]*|("?include"?\s+)(?:"([^"\r\n]+)"|([^\s{}]+))')
    output = {path: content for path, content in assets.items() if not path.endswith('.vmt')}
    for path, target in mapping.items():
        text = assets[path].decode('utf8', errors='replace')
        def rewrite(match):
            if match.group(1) is None:
                return match.group(0)  # retain comments without resolving them
            original = (match.group(2) or match.group(3)).replace('\\', '/').lower()
            original = 'materials/' + original.removeprefix('materials/').removesuffix('.vmt') + '.vmt'
            if original not in mapping:
                raise ValueError(f'Falta dependencia include del charm: {original}')
            return match.group(1) + _quote('materials/' + mapping[original] + '.vmt')
        output['materials/' + target + '.vmt'] = include.sub(rewrite, text).encode('utf8')
    remapped = [mapping['materials/' + name.replace('\\', '/').lower() + '.vmt'] for name in names]
    return remapped, output


def export_native_adapter(placement, charm_model: Model, charm_read, output_vpk,
                          *, compiler, workspace=None, timeout=120):
    """Compile and validate a charm-only VPK; return its runtime model path.

    ``charm_read`` is a Mount.read or Resources.reader callable resolving the
    charm's VMT/VTF dependencies. ``compiler`` is an explicit studiomdl.exe path.
    ``workspace`` defaults to the application workspace; build files and compiler
    logs are retained in its .build/native_adapter directory for diagnostics.
    The output is written only after the compiler and MDL round trip succeed.
    No weapon reader is accepted or needed. Changing placement transforms or
    target bone does not change the model path: native console commands set them.
    """
    placement.validate()
    if charm_model.path != placement.charm.lower() or digest(charm_model.data) != placement.charm_sha256:
        raise ValueError('El charm cargado no coincide con la huella del perfil')
    if not len(charm_model.triangles):
        raise ValueError('El charm no contiene geometría exportable')
    if not 1 <= len(charm_model.bones) <= 254:
        raise ValueError('El charm necesita entre 1 y 254 huesos originales')
    if any(bone.name in ('CS_native_root', 'CS_mount') for bone in charm_model.bones):
        raise ValueError('El charm ya usa nombres reservados del adaptador')
    jiggles = _jiggle_payloads(charm_model)
    compiler = Path(compiler).resolve()
    if not compiler.is_file():
        raise FileNotFoundError('Instala Left 4 Dead 2 Authoring Tools y selecciona bin/studiomdl.exe')
    game_install = compiler.parent.parent
    game_bin = game_install / 'left4dead2/bin'
    if not game_bin.is_dir():
        raise ValueError('studiomdl.exe debe pertenecer a la instalación de L4D2 Authoring Tools')
    pose = reference_pose(charm_model, placement.charm_pose)
    vertices = reference_vertices(charm_model, placement.charm_pose)
    normals = skin_directions(charm_model, pose, charm_model.normals)
    if not all(np.isfinite(array).all() for array in (vertices, normals, charm_model.uvs)):
        raise ValueError('La geometría del charm contiene valores no finitos')
    material_names, assets = _materials(charm_model.data, charm_read)
    identity = hashlib.sha256(b'CharmStudio native skinned adapter v4 l4d2-jiggle120\0')
    identity.update(charm_model.data)  # skeleton, weights' names and jiggle parameters
    identity.update(np.asarray(charm_model.bone_ids, dtype='<i4').tobytes())
    identity.update(np.asarray(charm_model.weights, dtype='<f8').tobytes())
    for array in (vertices, normals, charm_model.uvs):
        identity.update(np.asarray(array, dtype='<f8').tobytes())
    identity.update(np.asarray(charm_model.triangles, dtype='<i4').tobytes())
    identity.update(np.asarray(charm_model.material_ids, dtype='<i4').tobytes())
    identity.update('\0'.join(material_names).encode('utf8'))
    for path, content in sorted(assets.items()):
        if not path.startswith('materials/'):
            raise ValueError('Solo se admiten dependencias de materiales del charm')
        identity.update(path.encode('utf8') + b'\0' + content)
    model_path = 'models/charms_studio/native/p_' + identity.hexdigest()[:24] + '.mdl'
    # Compute the content identity before remapping: namespaced paths must not
    # feed back into their own hash. Keep VTF paths; only the repaired material
    # definitions need isolation from the original subscribed charm's VMTs.
    material_names, assets = _namespace_materials(material_names, assets, identity.hexdigest()[:24])
    build_root = Path(workspace or Path(__file__).resolve().parents[1]).resolve() / '.build/native_adapter'
    build_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=identity.hexdigest()[:12]+'_', dir=build_root))
    game = work / 'game'
    game.mkdir()
    (game / 'gameinfo.txt').write_text(
        '"GameInfo" { game "Charm Studio Native Build" FileSystem { '
        'SteamAppId 550 ToolsAppId 563 SearchPaths { '
        f'GameBin {_quote(game_bin)} Game {_quote(game)} '
        f'Game {_quote(game_install / "left4dead2")} }} }} }}\n', encoding='utf8')
    nodes = ['0 "CS_native_root" -1', '1 "CS_mount" 0']
    frames = ['0 0 0 0 0 0 0', '1 0 0 0 0 0 0']
    for index, bone in enumerate(charm_model.bones):
        parent = bone.parent+2 if bone.parent >= 0 else 1
        local = np.linalg.inv(pose[bone.parent]) @ pose[index] if bone.parent >= 0 else pose[index]
        nodes.append(f'{index+2} {_quote(bone.name)} {parent}')
        frames.append(f'{index+2} {_numbers(local[:3, 3])} {_numbers(_bone_euler(local))}')
    skeleton = ['version 1', 'nodes', *nodes, 'end', 'skeleton', 'time 0', *frames, 'end']
    smd = skeleton + ['triangles']
    for triangle, material in zip(charm_model.triangles, charm_model.material_ids):
        name = material_names[int(material)]
        _quote(name)  # reject malformed names before writing the line format
        smd.append(name)
        # studiomdl reverses SMD winding when generating VTX. Round-trip
        # evidence: all 3928 marraqueta faces reversed without this swap.
        for index in triangle[[0, 2, 1]]:
            uv = charm_model.uvs[index].copy()
            uv[1] = 1 - uv[1]
            links = [(int(bone)+2, float(weight)) for bone, weight in
                     zip(charm_model.bone_ids[index], charm_model.weights[index]) if weight > 0]
            if not links:
                links = [(2, 1.)]
            total = sum(weight for _, weight in links)
            smd.append(f'{links[0][0]} {_numbers(vertices[index])} {_numbers(normals[index])} {_numbers(uv)} {len(links)} ' +
                       ' '.join(f'{bone} {weight/total:.9f}' for bone, weight in links))
    (work / 'reference.smd').write_text('\n'.join(smd+['end'])+'\n', encoding='utf8')
    (work / 'idle.smd').write_text('\n'.join(skeleton)+'\n', encoding='utf8')
    qc = [f'$modelname {_quote(model_path.removeprefix("models/"))}', '$upaxis Z',
          '$origin 0 0 0 0', '$scale 1', '$body "charm" "reference.smd"',
          '$cdmaterials ""', '$surfaceprop "metal"', '$bonemerge "CS_native_root"',
          '$bonemerge "CS_mount"', '$sequence "idle" "idle.smd" fps 30 loop']
    qc.extend('$bonemerge '+_quote(bone.name) for bone in charm_model.bones)
    (work / 'adapter.qc').write_text('\n'.join(qc)+'\n', encoding='utf8')
    try:
        result = subprocess.run([str(compiler), '-game', str(game), str(work / 'adapter.qc')],
                                cwd=work, capture_output=True, text=True, errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired as error:
        (work / 'compiler.log').write_text(str(error), encoding='utf8')
        raise RuntimeError(f'studiomdl excedió el tiempo límite. Registro: {work / "compiler.log"}') from error
    (work / 'compiler.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf8')
    if result.returncode:
        raise RuntimeError(f'Error de studiomdl ({result.returncode}). Registro: {work / "compiler.log"}')
    stem = game / model_path.removesuffix('.mdl')
    files = dict(assets)
    for extension in ('.mdl', '.vvd', '.dx90.vtx', '.dx80.vtx', '.sw.vtx'):
        file = stem.with_name(stem.name+extension)
        if file.is_file():
            files[model_path.removesuffix('.mdl')+extension] = file.read_bytes()
    for extension in ('.mdl', '.vvd', '.dx90.vtx'):
        if model_path.removesuffix('.mdl')+extension not in files:
            raise RuntimeError(f'studiomdl no generó {extension}. Registro: {work / "compiler.log"}')
    files[model_path] = _restore_jiggles(files[model_path], jiggles)
    compiled = Model(files.__getitem__, model_path)
    if ([bone.name for bone in compiled.bones] != ['CS_native_root', 'CS_mount']+[bone.name for bone in charm_model.bones] or
            compiled.bones[1].parent != 0 or len(compiled.triangles) != len(charm_model.triangles)):
        raise ValueError('El modelo compilado no conserva el esqueleto o los triángulos del charm')
    if np.any(compiled.bone_ids[compiled.weights > 0] < 2):
        raise ValueError('El modelo compilado contiene pesos fuera de los huesos originales')
    output = Path(output_vpk).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = work / 'validated.vpk'
    write_vpk(staged, files)
    archive = VPK(staged)
    Model(archive.read, model_path)
    for path in archive.entries:
        if not (path.startswith('materials/') or path.startswith(model_path.removesuffix('.mdl')+'.')):
            raise ValueError('El VPK contiene recursos ajenos al charm')
        archive.read(path)  # verify CRC for every packaged resource
    output.write_bytes(staged.read_bytes())
    return NativeAdapterExport(model_path, output, work, tuple(sorted(files)))
