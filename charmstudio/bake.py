"""Bake a rigid charm into a compiled Source viewmodel without recompiling animation.

Only MDL 48/49 + VVD 4 + VTX 7 are accepted. The original model descriptors,
animation blocks, bones, events and strips stay at their original byte addresses.
New descriptor arrays point back to them and add a final always-on bodypart.
The exported model uses LOD0 only. No attachment or server entity is involved.
"""
from __future__ import annotations

import hashlib
import math
import re
import struct
import numpy as np
from .materials import repair_material
from .charm_pose import reference_pose, skin_directions
from .source import Model


def _i(data, offset):
    return struct.unpack_from('<i', data, offset)[0]


def _str(data, offset):
    if offset < 0 or offset >= len(data):
        raise ValueError('Cadena MDL fuera del archivo')
    end = data.find(b'\0', offset)
    if end < 0:
        raise ValueError('Cadena MDL truncada')
    return data[offset:end].decode('utf8', errors='replace').replace('\\', '/')


def _put(data, offset, value):
    struct.pack_into('<i', data, offset, value)


def _alloc(data, count):
    data.extend(b'\0' * (-len(data) % 4))
    offset = len(data)
    data.extend(b'\0' * count)
    return offset


def _cstring(data, value):
    offset = len(data)
    data.extend(value.encode('utf8') + b'\0')
    return offset


def _checked_mdl(data):
    if len(data) < 408 or data[:4] != b'IDST' or _i(data, 4) not in (48, 49):
        raise ValueError('Integrar requiere MDL Source versión 48 o 49')
    if _i(data, 76) != len(data):
        raise ValueError('La longitud declarada del MDL no coincide')


def _vvd(data, checksum):
    if len(data) < 64 or data[:4] != b'IDSV' or _i(data, 4) != 4:
        raise ValueError('VVD inválido o no compatible')
    if _i(data, 8) != checksum:
        raise ValueError('MDL y VVD tienen checksums distintos')
    count = _i(data, 16)
    if count < 0 or count > 4_000_000:
        raise ValueError('Demasiados vértices VVD')
    nfix, fixes, vertices, tangents = struct.unpack_from('<4i', data, 48)
    if vertices < 64 or vertices + count * 48 > len(data):
        raise ValueError('Vértices VVD truncados')
    if tangents <= 0 or tangents + count * 16 > len(data):
        raise ValueError('La exportación necesita tangentes VVD válidas')
    order = []
    if nfix:
        if nfix < 0 or fixes < 64 or fixes + nfix * 12 > len(data):
            raise ValueError('Tabla de fixups VVD inválida')
        for j in range(nfix):
            lod, start, size = struct.unpack_from('<3i', data, fixes + j * 12)
            if start < 0 or size < 0 or start + size > count:
                raise ValueError('Fixup VVD fuera de rango')
            if lod >= 0:
                order.extend(range(start, start + size))
    else:
        order = list(range(count))
    if len(order) != count:
        raise ValueError('Los fixups VVD no reconstruyen LOD0 completo')
    return (b''.join(data[vertices+j*48:vertices+(j+1)*48] for j in order),
            b''.join(data[tangents+j*16:tangents+(j+1)*16] for j in order))


def _bodyparts(data):
    n, off = struct.unpack_from('<2i', data, 232)
    if n < 0 or n > 1024 or off < 0 or off + n * 16 > len(data):
        raise ValueError('Bodyparts MDL inválidos')
    return [off + j * 16 for j in range(n)]


def _vtx_meshes(data, mdl_version, checksum):
    if len(data) < 36 or _i(data, 0) != 7 or _i(data, 16) != checksum:
        raise ValueError('VTX inválido o checksum distinto del MDL')
    bp_count, bp_off = struct.unpack_from('<2i', data, 28)
    result = []
    for bi in range(bp_count):
        bp = bp_off + bi * 8
        models, rel = struct.unpack_from('<2i', data, bp)
        if models < 1:
            result.append([])
            continue
        model = bp + rel  # first bodygroup option only
        lods, rel = struct.unpack_from('<2i', data, model)
        if lods < 1:
            raise ValueError('VTX no contiene LOD0')
        lod = model + rel
        mesh_count, rel = struct.unpack_from('<2i', data, lod)
        meshes = []
        for mi in range(mesh_count):
            mesh = lod + rel + mi * 9
            groups, grouprel = struct.unpack_from('<2i', data, mesh)
            triangles = []
            for gi in range(groups):
                sg = mesh + grouprel + gi * (33 if mdl_version == 49 else 25)
                nv, vo, ni, io, ns, so = struct.unpack_from('<6i', data, sg)
                if not (0 <= nv <= 65535 and 0 <= ni <= 10_000_000 and 0 <= ns <= 100000):
                    raise ValueError('Stripgroup VTX inválido')
                verts = [struct.unpack_from('<H', data, sg + vo + j * 9 + 4)[0] for j in range(nv)]
                indices = struct.unpack_from(f'<{ni}H', data, sg + io)
                for si in range(ns):
                    strip = sg + so + si * (35 if mdl_version == 49 else 27)
                    count, start = struct.unpack_from('<2i', data, strip)
                    flags = data[strip + 18]
                    if start < 0 or count < 0 or start + count > ni:
                        raise ValueError('Índices strip VTX fuera de rango')
                    raw = indices[start:start+count]
                    if any(j >= nv for j in raw):
                        raise ValueError('Vértice strip VTX fuera de rango')
                    if flags & 1:
                        if count % 3:
                            raise ValueError('Lista triangular VTX incompleta')
                        triangles.extend(tuple(verts[j] for j in raw[k:k+3]) for k in range(0, count, 3))
                    elif flags & 2:
                        for k in range(count - 2):
                            t = [raw[k], raw[k+1], raw[k+2]]
                            if k & 1:
                                t[0], t[1] = t[1], t[0]
                            if len(set(t)) == 3:
                                triangles.append(tuple(verts[j] for j in t))
                    else:
                        raise ValueError('Tipo de strip VTX no compatible')
            meshes.append(triangles)
        result.append(meshes)
    return result


def _rotation(angles):
    pitch, yaw, roll = np.radians(angles)
    sp, cp, sy, cy, sr, cr = math.sin(pitch), math.cos(pitch), math.sin(yaw), math.cos(yaw), math.sin(roll), math.cos(roll)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
                     [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
                     [-sp, sr*cp, cr*cp]], dtype=float)


def _materials(data, read):
    nt, to, nd, do = struct.unpack_from('<4i', data, 204)
    dirs = [_str(data, _i(data, do + j * 4)) for j in range(nd)] or ['']
    names, files = [], {}
    visited = set()
    def dependency(path):
        path = path.replace('\\', '/').lower()
        if path in visited:
            return
        if path.startswith('/') or '..' in path.split('/') or ':' in path:
            raise ValueError('Ruta de material insegura')
        visited.add(path)
        try:
            content = read(path)
        except (FileNotFoundError, KeyError) as exc:
            raise ValueError(f'Falta dependencia del charm: {path}') from exc
        if path.endswith('.vmt'):
            content, _ = repair_material(path, content, read)
        files[path] = content
        if path.endswith('.vmt'):
            text = re.sub(r'//[^\n]*', '', content.decode('utf8', errors='replace'))
            tokens = re.findall(r'"([^"\r\n]*)"|([^\s{}"]+)', text)
            tokens = [a or b for a, b in tokens]
            for i, token in enumerate(tokens[:-1]):
                key, value = token.lower(), tokens[i+1].replace('\\', '/')
                if key == 'include':
                    dependency('materials/' + value.removeprefix('materials/').removesuffix('.vmt') + '.vmt')
                if key in {'$basetexture', '$basetexture2', '$bumpmap', '$bumpmap2', '$normalmap', '$detail', '$envmapmask', '$phongexponenttexture', '$lightwarptexture', '$selfillummask', '$blendmodulatetexture'}:
                    if value and not value.startswith(('_', '[', '{')):
                        dependency('materials/' + value.removeprefix('materials/').removesuffix('.vtf') + '.vtf')
    for j in range(nt):
        tex = to + j * 64
        name = _str(data, tex + _i(data, tex))
        found = None
        for directory in dirs:
            full = (directory.rstrip('/') + '/' + name).lstrip('/') if directory else name
            try:
                read('materials/' + full + '.vmt')
                found = full
                break
            except (FileNotFoundError, KeyError):
                pass
        if found is None:
            raise ValueError(f'No se encontró VMT de {name}; directorios: {dirs}')
        names.append(found)
        dependency('materials/' + found + '.vmt')
    return names, files


def _append_rigid_mesh(vtx, vertices, triangles, bone, mdl_version):
    """Return new VTX mesh header offset (one strip, one hardware bone)."""
    if vertices > 65535:
        raise ValueError('Una malla charm supera 65535 vértices')
    if any(v < 0 or v >= vertices for t in triangles for v in t):
        raise ValueError('Triángulo charm fuera de su malla')
    mesh = _alloc(vtx, 9)
    sg = _alloc(vtx, 33 if mdl_version == 49 else 25)
    strip = _alloc(vtx, 35 if mdl_version == 49 else 27)
    state = _alloc(vtx, 8)
    vo = _alloc(vtx, vertices * 9)
    io = _alloc(vtx, len(triangles) * 6)
    struct.pack_into('<2iB', vtx, mesh, 1, sg - mesh, 0)
    struct.pack_into('<6iB', vtx, sg, vertices, vo-sg, len(triangles)*3, io-sg, 1, strip-sg, 2)
    struct.pack_into('<4ihB2i', vtx, strip, len(triangles)*3, 0, vertices, 0, 1, 1, 1, state-strip)
    struct.pack_into('<2i', vtx, state, 0, bone)
    for j in range(vertices):
        struct.pack_into('<4BH3B', vtx, vo + j * 9, 0, 0, 0, 1, j, 0, 0, 0)
    for j, triangle in enumerate(triangles):
        struct.pack_into('<3H', vtx, io + j * 6, *triangle)
    return mesh


def bake(weapon_read, weapon_path, charm_read, charm_path, bone_index, offset, angles, scale, charm_pose='bind'):
    """Return an override's path->bytes map. Does not write or alter any input.

    Charm bodygroups use option zero. Charm animation/physics are deliberately
    frozen: every added vertex receives weight 1 on the selected weapon bone.
    The caller must package the map as a separate addon and disable conflicting
    overrides of this exact viewmodel. Runtime console transforms do not affect
    this baked model; export again after editing a placement.
    """
    params = np.asarray([*offset, *angles, scale], dtype=float)
    if params.shape != (7,) or not np.isfinite(params).all() or not 0 < scale <= 10:
        raise ValueError('Transformación no válida')
    weapon_path, charm_path = weapon_path.replace('\\', '/'), charm_path.replace('\\', '/')
    if not weapon_path.lower().endswith('.mdl') or not charm_path.lower().endswith('.mdl'):
        raise ValueError('Se requieren rutas MDL')
    wd, cd = weapon_read(weapon_path), charm_read(charm_path)
    _checked_mdl(wd)
    _checked_mdl(cd)
    version = _i(wd, 4)
    checksum = _i(wd, 8)
    nb, bo = struct.unpack_from('<2i', wd, 156)
    if not isinstance(bone_index, int) or not 0 <= bone_index < nb or bone_index > 127:
        raise ValueError('Hueso weapon fuera de rango (máximo 127 para este exportador)')
    wbase, cbase = weapon_path[:-4], charm_path[:-4]
    wvertices, wtangents = _vvd(weapon_read(wbase + '.vvd'), checksum)
    cvertices, ctangents = _vvd(charm_read(cbase + '.vvd'), _i(cd, 8))
    posed_positions = posed_normals = posed_tangents = None
    if charm_pose not in ('bind','sequence0'):
        raise ValueError('Pose del charm no admitida')
    if charm_pose != 'bind':
        charm_model = Model(charm_read,charm_path)
        pose = reference_pose(charm_model,charm_pose)
        posed_positions = charm_model.skinned_vertices(pose)
        posed_normals = skin_directions(charm_model,pose,charm_model.normals)
        tangent_array = np.frombuffer(ctangents,dtype='<f4').reshape(-1,4)[:,:3]
        posed_tangents = skin_directions(charm_model,pose,tangent_array)
        if len(posed_positions)*48 != len(cvertices):
            raise ValueError('La pose del charm no coincide con su buffer VVD')
    wvtx = bytearray(weapon_read(wbase + '.dx90.vtx'))
    if len(wvtx) < 36 or _i(wvtx, 0) != 7 or _i(wvtx, 16) != checksum:
        raise ValueError('VTX weapon incompatible')
    charm_mesh_triangles = _vtx_meshes(charm_read(cbase + '.dx90.vtx'), _i(cd, 4), _i(cd, 8))
    material_names, files = _materials(cd, charm_read)
    mdl = bytearray(wd)
    # This output has one visible LOD, never a dedicated shadow-only LOD.
    hdr_flags = _i(wd, 152) & ~(0x40 | 0x100)
    if any(re.search(rb'(?i)"?\$translucent"?\s+"?1', value) for key, value in files.items() if key.endswith('.vmt')):
        hdr_flags = (hdr_flags & ~0x4) | 0x8  # retain a translucent material pass
    _put(mdl, 152, hdr_flags)
    wcount = len(wvertices) // 48
    vertices, tangents, meshes = bytearray(), bytearray(), []
    # poseToBone is inverse bind transform; invert to place new mesh in bind space.
    ptb = np.eye(4)
    ptb[:3] = np.asarray(struct.unpack_from('<12f', wd, bo + bone_index * 216 + 96)).reshape(3, 4)
    transform = np.linalg.inv(ptb)
    rotate = transform[:3, :3] @ _rotation(angles)
    translate = transform[:3, :3] @ np.asarray(offset) + transform[:3, 3]
    charm_skin_count, charm_family_count, charm_skin_offset = struct.unpack_from('<3i', cd, 220)
    charm_skin = struct.unpack_from(f'<{charm_skin_count}H', cd, charm_skin_offset) if charm_family_count else ()
    for bi, bp in enumerate(_bodyparts(cd)):
        nmodel, _, rel = struct.unpack_from('<3i', cd, bp + 4)
        if not nmodel:
            continue
        model = bp + rel
        nm, meshrel, nv, vertexbyte = struct.unpack_from('<4i', cd, model + 72)
        if bi >= len(charm_mesh_triangles) or nm != len(charm_mesh_triangles[bi]):
            raise ValueError('MDL y VTX charm tienen distintas mallas')
        for mi in range(nm):
            mesh = model + meshrel + mi * 116
            material, _, count, vo = struct.unpack_from('<4i', cd, mesh)
            if material < 0:
                raise ValueError('Material charm inválido')
            if charm_skin:
                if material >= len(charm_skin):
                    raise ValueError('Skin charm fuera de rango')
                material = charm_skin[material]
            if material >= len(material_names):
                raise ValueError('Material charm fuera de rango')
            start = vertexbyte // 48 + vo
            if start < 0 or count < 0 or (start + count)*48 > len(cvertices):
                raise ValueError('Vértices charm fuera de rango')
            new_offset = len(vertices) // 48
            for j in range(start, start + count):
                vertex = bytearray(cvertices[j*48:(j+1)*48])
                tangent = bytearray(ctangents[j*16:(j+1)*16])
                pos = posed_positions[j] if posed_positions is not None else np.asarray(struct.unpack_from('<3f', vertex, 16))
                normal = posed_normals[j] if posed_normals is not None else np.asarray(struct.unpack_from('<3f', vertex, 28))
                tangentxyz = posed_tangents[j] if posed_tangents is not None else np.asarray(struct.unpack_from('<3f', tangent))
                struct.pack_into('<3f4B', vertex, 0, 1, 0, 0, bone_index, 0, 0, 1)
                struct.pack_into('<3f', vertex, 16, *(rotate @ (pos*scale) + translate))
                struct.pack_into('<3f', vertex, 28, *(rotate @ normal))
                struct.pack_into('<3f', tangent, 0, *(rotate @ tangentxyz))
                vertices.extend(vertex)
                tangents.extend(tangent)
            meshes.append((material, new_offset, count, charm_mesh_triangles[bi][mi]))
    if not meshes or not vertices:
        raise ValueError('El charm no contiene geometría en bodygroup cero')
    if len(vertices)//48 > 65535:
        raise ValueError('El charm completo supera 65535 vértices')

    # Preserve old materials and add root-qualified charm materials.
    oldnt, oldto, oldnd, olddo = struct.unpack_from('<4i', wd, 204)
    nto = _alloc(mdl, (oldnt + len(material_names))*64)
    for j in range(oldnt):
        old, new = oldto+j*64, nto+j*64
        mdl[new:new+64] = wd[old:old+64]
        _put(mdl, new, old+_i(wd, old)-new)
    for j, name in enumerate(material_names):
        tex = nto + (oldnt+j)*64
        _put(mdl, tex, _cstring(mdl, name)-tex)
    ndo = _alloc(mdl, (oldnd+1)*4)
    for j in range(oldnd):
        _put(mdl, ndo+j*4, _i(wd, olddo+j*4))
    _put(mdl, ndo+oldnd*4, _cstring(mdl, ''))
    struct.pack_into('<4i', mdl, 204, oldnt+len(material_names), nto, oldnd+1, ndo)
    skinrefs, families, skinpos = struct.unpack_from('<3i', wd, 220)
    if skinrefs == 0 or families == 0:
        skinrefs, families = oldnt, 1
        skins = list(range(oldnt))
    else:
        skins = struct.unpack_from(f'<{skinrefs*families}H', wd, skinpos)
    newskin = _alloc(mdl, (skinrefs+len(material_names))*families*2)
    for family in range(families):
        row = list(skins[family*skinrefs:(family+1)*skinrefs]) + list(range(oldnt, oldnt+len(material_names)))
        struct.pack_into(f'<{len(row)}H', mdl, newskin+family*len(row)*2, *row)
    struct.pack_into('<3i', mdl, 220, skinrefs+len(material_names), families, newskin)

    oldbodies = _bodyparts(wd)
    bodies = _alloc(mdl, (len(oldbodies)+1)*16)
    for j, old in enumerate(oldbodies):
        new = bodies+j*16
        mdl[new:new+16] = wd[old:old+16]
        _put(mdl, new, old+_i(wd, old)-new)
        _put(mdl, new+12, old+_i(wd, old+12)-new)
        # Only LOD0 is exported; no mesh should be recalculated for lower LODs.
        for option in range(_i(wd, old+4)):
            model = old+_i(wd, old+12)+option*148
            for mi in range(_i(wd, model+72)):
                mesh = model+_i(wd, model+76)+mi*116
                struct.pack_into('<8i', mdl, mesh+52, *([_i(wd, mesh+8)]*8))
    bp = bodies+len(oldbodies)*16
    model = _alloc(mdl, 148)
    mesharray = _alloc(mdl, len(meshes)*116)
    name = _cstring(mdl, 'charm_studio')
    struct.pack_into('<4i', mdl, bp, name-bp, 1, 1, model-bp)
    mdl[model:model+12] = b'charm_studio'
    struct.pack_into('<f', mdl, model+68, float(np.linalg.norm(np.ptp(np.asarray([struct.unpack_from('<3f', vertices, j+16) for j in range(0, len(vertices), 48)]), axis=0))))
    struct.pack_into('<5i', mdl, model+72, len(meshes), mesharray-model, len(vertices)//48, wcount*48, wcount*16)
    meshid = 0
    for old in oldbodies:
        for option in range(_i(wd, old+4)):
            om = old+_i(wd, old+12)+option*148
            meshid += _i(wd, om+72)
    for j, (material, vo, count, triangles) in enumerate(meshes):
        mesh = mesharray+j*116
        struct.pack_into('<4i', mdl, mesh, skinrefs+material, model-mesh, count, vo)
        _put(mdl, mesh+32, meshid+j)
        struct.pack_into('<8i', mdl, mesh+52, *([count]*8))
    struct.pack_into('<2i', mdl, 232, len(oldbodies)+1, bodies)
    added_positions = np.asarray([struct.unpack_from('<3f', vertices, j+16) for j in range(0, len(vertices), 48)])
    # Keep a charm just outside the original weapon bounds from being culled.
    view_min = np.minimum(np.asarray(struct.unpack_from('<3f', wd, 128)), added_positions.min(axis=0))
    view_max = np.maximum(np.asarray(struct.unpack_from('<3f', wd, 140)), added_positions.max(axis=0))
    struct.pack_into('<3f', mdl, 128, *view_min)
    struct.pack_into('<3f', mdl, 140, *view_max)
    mdl[377:379] = bytes((0, 1))
    # Ensure the engine actually evaluates the selected bone and its ancestors.
    bone = bone_index
    seen = set()
    linear_flags = None
    hdr2 = _i(wd, 400)
    if hdr2:
        if hdr2 < 0 or hdr2+20 > len(wd):
            raise ValueError('Cabecera secundaria MDL fuera de rango')
        linear_rel = _i(wd, hdr2+16)
        if linear_rel:
            linear = hdr2+linear_rel
            if _i(wd, linear) != nb:
                raise ValueError('Tabla lineal de huesos incompatible')
            linear_flags = linear+_i(wd, linear+4)
            if linear_flags < 0 or linear_flags+nb*4 > len(wd):
                raise ValueError('Flags lineales de huesos fuera de rango')
    while bone >= 0:
        if bone in seen or bone >= nb:
            raise ValueError('Jerarquía de huesos circular o fuera de rango')
        seen.add(bone)
        boneoff = bo+bone*216
        _put(mdl, boneoff+160, _i(mdl, boneoff+160) | 0x400)
        if linear_flags is not None:
            _put(mdl, linear_flags+bone*4, _i(mdl, linear_flags+bone*4) | 0x400)
        bone = _i(mdl, boneoff+4)

    # Original VTX subtrees are retained; copy only bodypart descriptors.
    nbp, obp = struct.unpack_from('<2i', wvtx, 28)
    if nbp != len(oldbodies):
        raise ValueError('MDL y VTX weapon tienen distintos bodyparts')
    newbp = _alloc(wvtx, (nbp+1)*8)
    for j in range(nbp):
        old, new = obp+j*8, newbp+j*8
        nmodels, rel = struct.unpack_from('<2i', wvtx, old)
        struct.pack_into('<2i', wvtx, new, nmodels, old+rel-new)
        for option in range(nmodels):
            _put(wvtx, old+rel+option*8, 1)
    bp = newbp+nbp*8
    vm = _alloc(wvtx, 8)
    lod = _alloc(wvtx, 12)
    vmeshes = _alloc(wvtx, len(meshes)*9)
    struct.pack_into('<2i', wvtx, bp, 1, vm-bp)
    struct.pack_into('<2i', wvtx, vm, 1, lod-vm)
    struct.pack_into('<2if', wvtx, lod, len(meshes), vmeshes-lod, 0)
    for j, (_, _, count, triangles) in enumerate(meshes):
        original = _append_rigid_mesh(wvtx, count, triangles, bone_index, version)
        new = vmeshes+j*9
        groups, rel = struct.unpack_from('<2i', wvtx, original)
        struct.pack_into('<2iB', wvtx, new, groups, original+rel-new, 0)
    _put(wvtx, 20, 1)
    struct.pack_into('<2i', wvtx, 28, nbp+1, newbp)

    # New checksum binds all three model resources and invalidates stale caches.
    identity = hashlib.sha256(wd + cd + params.astype('<f8').tobytes() + struct.pack('<i', bone_index) + charm_pose.encode('ascii')).digest()
    newchecksum = struct.unpack('<i', identity[:4])[0]
    _put(mdl, 8, newchecksum)
    _put(mdl, 76, len(mdl))
    _put(wvtx, 16, newchecksum)
    allvertices, alltangents = wvertices+vertices, wtangents+tangents
    total = len(allvertices)//48
    vvd = struct.pack('<4s3i8i4i', b'IDSV', 4, newchecksum, 1, *([total]*8), 0, 64, 64, 64+len(allvertices)) + allvertices + alltangents
    files.update({weapon_path: bytes(mdl), wbase+'.vvd': bytes(vvd), wbase+'.dx90.vtx': bytes(wvtx)})
    return files
