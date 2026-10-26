"""Experimental normal-addon export: survivor arms plus a bone-bound charm.

Requires L4D2 Authoring Tools. It replaces the single CharacterViewmodelAddon
model; it does not copy the weapon mesh or install anything into the game.
Arms with skin families or bodygroup alternatives are rejected in this initial
prototype. Charm bodygroups/skins use option zero; its chosen pose becomes idle.
Arms materials remain external, supplied by the mounted game/survivor addons.
Placement changes require exporting another model; no live console API is added.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .bake import _materials
from .charm_pose import reference_pose, reference_vertices, skin_directions
from .native_adapter import (_bone_euler, _jiggle_payloads, _namespace_materials,
                             _numbers, _quote, _restore_jiggles, L4D2_JIGGLE)
from .project import digest
from .resources import normalize
from .source import Model
from .transforms import transform, points


def _arms_material_names(model, read):
    result = []
    for name in model.materials:
        for directory in model.material_paths or ['']:
            candidate = normalize((directory.rstrip('/\\')+'/'+name).lstrip('/\\'))
            try:
                read('materials/'+candidate+'.vmt')
                result.append(candidate)
                break
            except (FileNotFoundError, KeyError):
                pass
        else:
            raise ValueError(f'Falta material externo de brazos: {name}')
    return result


def _scaled_jiggle(payload, scale):
    # L4D2 layout: flags at slot 0, distances at these 32-bit field indices.
    values = list(L4D2_JIGGLE.unpack(payload))
    for index in (1, 21, 22, 24, 25, 27, 28):
        values[index] *= scale
    return L4D2_JIGGLE.pack(*values)


def export_natural_arms(arms_model, arms_read, charm_model, charm_read, profile,
                        output_modelpath, compiler, workspace, *, timeout=120):
    """Return compiled model sidecars + isolated charm materials as path->bytes.

    Build files and compiler.log stay under workspace/.build/natural_arms.
    output_modelpath must be a unique models/charms_studio/natural/...mdl path
    shorter than the installed factory's 64-byte cache. Caller packages files
    and explicitly sets CharacterViewmodelAddon in its own weapon-script addon.
    """
    profile.validate()
    output_modelpath = normalize(output_modelpath)
    if (not output_modelpath.startswith('models/charms_studio/natural/') or
            not output_modelpath.endswith('.mdl') or len(output_modelpath.encode('ascii')) >= 64):
        raise ValueError('La ruta natural debe estar en models/charms_studio/natural y ocupar menos de 64 bytes')
    if charm_model.path != profile.charm.lower() or digest(charm_model.data) != profile.charm_sha256:
        raise ValueError('El charm cargado no coincide con la huella del perfil')
    if not len(arms_model.triangles) or not len(charm_model.triangles):
        raise ValueError('Brazos y charm deben contener geometría')
    if arms_model.r.i(224) > 1:
        raise ValueError('Este prototipo no admite familias de skins en brazos')
    bodyparts, offset = arms_model.r.unpack('ii', 232)
    if any(arms_model.r.i(offset+index*16+4) > 1 for index in range(bodyparts)):
        raise ValueError('Este prototipo no admite variantes de bodygroups en brazos')
    compiler = Path(compiler).resolve()
    if not compiler.is_file():
        raise FileNotFoundError('Instala L4D2 Authoring Tools y selecciona bin/studiomdl.exe')
    install = compiler.parent.parent
    if not (install/'left4dead2/bin').is_dir():
        raise ValueError('El compilador debe pertenecer a L4D2 Authoring Tools')

    arms_pose = arms_model.bind_pose()
    charm_pose = reference_pose(charm_model, profile.charm_pose)
    arm_names = [bone.name for bone in arms_model.bones]
    if len(set(arm_names)) != len(arm_names):
        raise ValueError('Los brazos tienen nombres de hueso duplicados')
    names = arm_names.copy()
    parents = [bone.parent for bone in arms_model.bones]
    local_matrices = [np.linalg.inv(arms_pose[bone.parent])@arms_pose[index] if bone.parent >= 0
                      else arms_pose[index] for index, bone in enumerate(arms_model.bones)]
    if profile.bone in names:
        target = names.index(profile.bone)
        anchor = arms_pose[target]
    else:
        target = len(names)
        names.append(profile.bone)
        parents.append(-1)
        local_matrices.append(np.eye(4))
        anchor = np.eye(4)
    merged_count = len(names)
    mount = len(names)
    if 'CSMount' in names:
        raise ValueError('Los brazos usan el nombre reservado CSMount')
    names.append('CSMount')
    parents.append(target)
    placement = transform(profile.offset, profile.angles, 1)
    local_matrices.append(placement)
    charm_start = len(names)
    for index, bone in enumerate(charm_model.bones):
        name = f'CSCharm_{index}_{bone.name}'
        if name in names:
            raise ValueError('Colisión de nombres en el esqueleto del charm')
        names.append(name)
        parents.append(charm_start+bone.parent if bone.parent >= 0 else mount)
        local = np.linalg.inv(charm_pose[bone.parent])@charm_pose[index] if bone.parent >= 0 else charm_pose[index].copy()
        local = local.copy()
        local[:3, 3] *= profile.scale
        local_matrices.append(local)
    if len(names) > 128:
        raise ValueError('El modelo compuesto supera el límite de 128 huesos del compilador L4D2')
    jiggles = dict(_jiggle_payloads(arms_model))
    source_jiggles = _jiggle_payloads(charm_model)
    for index, bone in enumerate(charm_model.bones):
        if bone.name in source_jiggles:
            jiggles[names[charm_start+index]] = _scaled_jiggle(source_jiggles[bone.name], profile.scale)

    charm_names, assets = _materials(charm_model.data, charm_read)
    identity = hashlib.sha256(b'NaturalArms prototype v1\0'+arms_model.data+charm_model.data+
        np.asarray([*profile.offset, *profile.angles, profile.scale], dtype='<f8').tobytes()+
        profile.bone.encode('ascii')+profile.charm_pose.encode('ascii')).hexdigest()[:24]
    charm_names, assets = _namespace_materials(charm_names, assets, 'natural_'+identity)
    arms_materials = _arms_material_names(arms_model, arms_read)
    charm_transform = anchor@placement
    charm_vertices = points(charm_transform, reference_vertices(charm_model, profile.charm_pose)*profile.scale)
    charm_normals = skin_directions(charm_model, charm_pose, charm_model.normals)@charm_transform[:3, :3].T
    if not all(np.isfinite(array).all() for array in (charm_vertices, charm_normals)):
        raise ValueError('La pose del charm contiene geometría no finita')
    nodes = [f'{index} {_quote(name)} {parent}' for index, (name, parent) in enumerate(zip(names, parents))]
    frames = [f'{index} {_numbers(matrix[:3, 3])} {_numbers(_bone_euler(matrix))}'
              for index, matrix in enumerate(local_matrices)]
    skeleton = ['version 1', 'nodes', *nodes, 'end', 'skeleton', 'time 0', *frames, 'end']
    smd = skeleton+['triangles']
    for model, vertices, normals, materials, shift in (
            (arms_model, arms_model.positions, arms_model.normals, arms_materials, 0),
            (charm_model, charm_vertices, charm_normals, charm_names, charm_start)):
        for triangle, material in zip(model.triangles, model.material_ids):
            smd.append(materials[int(material)])
            for index in triangle[[0, 2, 1]]:
                uv = model.uvs[index].copy()
                uv[1] = 1-uv[1]
                links = [(int(bone)+shift, float(weight)) for bone, weight in
                         zip(model.bone_ids[index], model.weights[index]) if weight > 0]
                if not links:
                    raise ValueError('El modelo contiene vértices sin pesos')
                total = sum(weight for _, weight in links)
                smd.append(f'{links[0][0]} {_numbers(vertices[index])} {_numbers(normals[index])} {_numbers(uv)} {len(links)} '+
                           ' '.join(f'{bone} {weight/total:.9f}' for bone, weight in links))

    build_root = Path(workspace).resolve()/'.build/natural_arms'
    build_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=identity[:12]+'_', dir=build_root))
    game = work/'game'
    game.mkdir()
    (game/'gameinfo.txt').write_text('"GameInfo" { game "Natural Arms Lab" FileSystem { SteamAppId 550 ToolsAppId 563 SearchPaths { '+
        f'GameBin {_quote(install/"left4dead2/bin")} Game {_quote(game)} Game {_quote(install/"left4dead2")} }} }} }}\n', encoding='utf8')
    (work/'reference.smd').write_text('\n'.join(smd+['end'])+'\n', encoding='utf8')
    (work/'idle.smd').write_text('\n'.join(skeleton)+'\n', encoding='utf8')
    qc = [f'$modelname {_quote(output_modelpath.removeprefix("models/"))}', '$upaxis Z', '$origin 0 0 0 0',
          '$scale 1', '$body "arms_charm" "reference.smd"', '$cdmaterials ""', '$surfaceprop "flesh"',
          '$sequence "idle" "idle.smd" fps 30 loop']
    qc += ['$bonemerge '+_quote(name) for name in names[:merged_count]]
    # CSMount and the prefixed charm bones must retain their own local pose;
    # do not bonemerge them into the weapon's skeleton.
    (work/'adapter.qc').write_text('\n'.join(qc)+'\n', encoding='utf8')
    try:
        result = subprocess.run([str(compiler), '-game', str(game), str(work/'adapter.qc')],
                                cwd=work, capture_output=True, text=True, errors='replace', timeout=timeout)
    except subprocess.TimeoutExpired as error:
        (work/'compiler.log').write_text(str(error), encoding='utf8')
        raise RuntimeError(f'studiomdl excedió el tiempo límite: {work/"compiler.log"}') from error
    (work/'compiler.log').write_text(result.stdout+'\n'+result.stderr, encoding='utf8')
    if result.returncode:
        raise RuntimeError(f'Error de studiomdl ({result.returncode}): {work/"compiler.log"}')
    files = dict(assets)
    for extension in ('.mdl', '.vvd', '.dx90.vtx', '.dx80.vtx', '.sw.vtx'):
        path = output_modelpath.removesuffix('.mdl')+extension
        file = game/path
        if file.is_file():
            files[path] = file.read_bytes()
    if any(output_modelpath.removesuffix('.mdl')+extension not in files for extension in ('.mdl', '.vvd', '.dx90.vtx')):
        raise RuntimeError(f'El compilador no generó todos los sidecars: {work/"compiler.log"}')
    files[output_modelpath] = _restore_jiggles(files[output_modelpath], jiggles)
    compiled = Model(files.__getitem__, output_modelpath)
    if [bone.name for bone in compiled.bones] != names:
        raise ValueError(f'El compilador cambió huesos del modelo natural: {work/"compiler.log"}')
    if len(compiled.triangles) != len(arms_model.triangles)+len(charm_model.triangles):
        raise ValueError('El compilador cambió el número de triángulos del compuesto')
    return files
