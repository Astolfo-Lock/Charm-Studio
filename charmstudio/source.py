"""Offline Source MDL 48/49 + VVD4 + VTX7 reader.

Layouts follow ValveSoftware/source-sdk-2013 src/public/{studio,optimize}.h.
Frame-major v49 tracks follow the published Reactive Drop SDK studio.h
(ReactiveDrop/reactivedrop_public_src, src/public/studio.h).
The preview evaluates stored skeletal tracks; it does not emulate engine IK,
jiggle physics, procedural bones, pose-parameter blends or animation layers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import struct
from pathlib import PurePosixPath
import numpy as np


class FormatError(ValueError):
    pass


class Reader:
    def __init__(self, data):
        self.data = data

    def unpack(self, fmt, offset):
        size = struct.calcsize('<' + fmt)
        if offset < 0 or offset + size > len(self.data):
            raise FormatError(f'Archivo Source truncado: offset {offset}, tamaño {size}')
        return struct.unpack_from('<' + fmt, self.data, offset)

    def i(self, offset):
        return self.unpack('i', offset)[0]

    def string(self, offset):
        if offset < 0 or offset >= len(self.data):
            raise FormatError('Cadena Source fuera del archivo')
        end = self.data.find(b'\0', offset, min(offset + 4096, len(self.data)))
        if end < 0:
            raise FormatError('Cadena Source sin terminador')
        return self.data[offset:end].decode('utf-8', 'replace')

    def table(self, offset, count, stride, limit=1000000):
        if count < 0 or count > limit or offset < 0 or offset + count * stride > len(self.data):
            raise FormatError('Tabla Source inválida o truncada')
        return range(count)


@dataclass
class Bone:
    name: str
    parent: int
    pos: np.ndarray
    quat: np.ndarray
    pose_to_bone: np.ndarray
    rot: np.ndarray
    posscale: np.ndarray
    rotscale: np.ndarray
    flags: int = 0


@dataclass
class Sequence:
    name: str
    fps: float
    frames: int
    animation_index: int
    supported: bool = True
    warning: str = ''
    owner: object = field(default=None, repr=False)


def quaternion_matrix(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-10 or not np.isfinite(norm):
        return np.eye(3)
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def euler_quaternion(e):
    x, y, z = np.asarray(e) / 2
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    return np.array([sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz,
                     cx*cy*sz-sx*sy*cz, cx*cy*cz+sx*sy*sz])


def quat48(data, offset):
    x, y, z = data.unpack('HHH', offset)
    v = np.array([(x-32768)/32768, (y-32768)/32768, ((z & 32767)-16384)/16384])
    return np.append(v, math.sqrt(max(0, 1-float(v@v))) * (-1 if z & 32768 else 1))


def quat64(data, offset):
    value = data.unpack('Q', offset)[0]
    v = np.array([(((value >> (21*i)) & 0x1fffff)-1048576)/1048576.5 for i in range(3)])
    return np.append(v, math.sqrt(max(0, 1-float(v@v))) * (-1 if value >> 63 else 1))


def rle_value(data, offset, frame):
    """Decode Source's valid/total signed-short runs (including held tails)."""
    for _ in range(100000):
        valid, total = data.unpack('BB', offset)
        if total == 0 or valid > total or valid == 0:
            raise FormatError('Run RLE de animación inválido')
        if frame < total:
            return data.unpack('h', offset + 2*(1+min(frame, valid-1)))[0]
        frame -= total
        offset += 2*(1+valid)
    raise FormatError('Animación RLE excede el límite')


class Model:
    def __init__(self, read, path, geometry=True, _visited=None):
        self.read, self.path = read, path.replace('\\', '/').lower()
        self.data = read(self.path)
        self.r = r = Reader(self.data)
        if self.data[:4] != b'IDST' or r.i(4) not in (48, 49):
            raise FormatError('Modelo incompatible: se requiere MDL Source versión 48/49')
        self.version, self.checksum = r.i(4), r.i(8)
        self.name = r.string(12)
        declared = r.i(76)
        if declared > len(self.data) or declared < 408:
            raise FormatError('Cabecera MDL truncada')
        self.warnings, self.bones, self.sequences, self.animations = [], [], [], []
        self.positions = np.empty((0, 3), dtype=float)
        self.weights = np.empty((0, 3), dtype=float)
        self.bone_ids = np.empty((0, 3), dtype=int)
        self.triangles = np.empty((0, 3), dtype=int)
        self.normals = np.empty((0, 3), dtype=float)
        self.uvs = np.empty((0, 2), dtype=float)
        self.material_ids = np.empty((0,), dtype=int)
        self.materials = []
        self._ani = None
        self._read_bones()
        self._read_animations()
        count, offset = r.unpack('ii', 204)
        for i in r.table(offset, count, 64, 4096):
            p = offset+i*64
            self.materials.append(r.string(p+r.i(p)))
        self.material_paths = []
        count, offset = r.unpack('ii', 212)
        for i in r.table(offset, count, 4, 4096):
            self.material_paths.append(r.string(r.i(offset+i*4)))
        if geometry:
            try:
                self._read_geometry()
            except (FileNotFoundError, KeyError) as error:
                self.warnings.append(f'Falta VVD/VTX: solo esqueleto ({error})')
        visited = set(_visited or ()) | {self.path}
        count, offset = r.unpack('ii', 336)
        for i in r.table(offset, count, 8, 256):
            p = offset+i*8
            child_path = r.string(p+r.i(p+4)).replace('\\', '/').lower()
            if not child_path.startswith('models/'):
                child_path = 'models/'+child_path
            if child_path in visited or len(visited) >= 16:
                self.warnings.append(f'Include cíclico o muy profundo: {child_path}')
                continue
            try:
                child = Model(read, child_path, geometry=False, _visited=visited)
                self.sequences.extend(child.sequences)
                self.warnings.extend(child.warnings)
            except (OSError, KeyError, ValueError) as error:
                self.warnings.append(f'Animaciones include no disponibles: {child_path}: {error}')

    def _read_bones(self):
        r = self.r
        count, offset = r.unpack('ii', 156)
        for i in r.table(offset, count, 216, 256):
            p = offset+i*216
            name, parent = r.unpack('ii', p)
            if parent < -1 or parent >= count or parent == i:
                raise FormatError('Padre de hueso inválido')
            matrix = np.eye(4)
            matrix[:3] = np.array(r.unpack('12f', p+96)).reshape(3, 4)
            self.bones.append(Bone(r.string(p+name), parent,
                np.array(r.unpack('3f', p+32)), np.array(r.unpack('4f', p+44)), matrix,
                np.array(r.unpack('3f', p+60)), np.array(r.unpack('3f', p+72)),
                np.array(r.unpack('3f', p+84)), r.i(p+160)))
            if r.i(p+164):
                self.warnings.append(f'Hueso procedural: {self.bones[-1].name}; preview sin física procedural')

    def _read_animations(self):
        r = self.r
        count, offset = r.unpack('ii', 180)
        for i in r.table(offset, count, 100, 16384):
            p = offset+i*100
            fps, flags, frames = r.unpack('fii', p+8)
            self.animations.append(dict(offset=p, name=r.string(p+r.i(p+4)), fps=fps,
                flags=flags, frames=max(1, frames), block=r.i(p+52), index=r.i(p+56),
                sections=r.i(p+80), sectionframes=r.i(p+84)))
        count, offset = r.unpack('ii', 188)
        for i in r.table(offset, count, 212, 16384):
            p = offset+i*212
            blends, blend_offset = r.unpack('ii', p+56)
            anim = r.unpack('h', p+blend_offset)[0] if blends else -1
            name = r.string(p+r.i(p+4))
            valid = 0 <= anim < len(self.animations)
            info = self.animations[anim] if valid else dict(fps=30, frames=1, flags=0)
            warnings = []
            if blends > 1:
                warnings.append('Preview usa la primera muestra del blend')
            if r.i(p+144) or r.i(p+148):
                warnings.append('Preview sin IK ni autolayers')
            if not valid:
                warnings.append('No hay pista local de animación')
            self.sequences.append(Sequence(name, info['fps'], info['frames'], anim, valid,
                '; '.join(warnings), self))

    def _read_geometry(self):
        stem = self.path[:-4]
        vvd = Reader(self.read(stem+'.vvd'))
        if vvd.data[:4] != b'IDSV' or vvd.i(4) != 4 or vvd.i(8) != self.checksum:
            raise FormatError('VVD incompatible o checksum diferente al MDL')
        count = vvd.i(16)
        fixups, fixstart, start = vvd.unpack('iii', 48)
        vvd.table(start, count, 48)
        dtype = np.dtype([('weights', '<f4', 3), ('bones', 'u1', 3), ('numbones', 'u1'),
                          ('pos', '<f4', 3), ('normal', '<f4', 3), ('uv', '<f4', 2)])
        vertices = np.frombuffer(vvd.data, dtype=dtype, count=count, offset=start)
        if fixups:
            order = []
            for i in vvd.table(fixstart, fixups, 12):
                lod, source, size = vvd.unpack('iii', fixstart+i*12)
                if source < 0 or size < 0 or source+size > count:
                    raise FormatError('VVD fixup fuera del buffer')
                if lod >= 0:
                    order.extend(range(source, source+size))
            if len(order) != count:
                raise FormatError('VVD fixups no reconstruyen LOD0')
            vertices = vertices[order]
        self.positions = vertices['pos'].astype(float)
        self.normals = vertices['normal'].astype(float)
        self.uvs = vertices['uv'].astype(float)
        self.weights = vertices['weights'].astype(float)
        self.bone_ids = vertices['bones'].astype(int)
        active = np.arange(3)[None, :] < vertices['numbones'][:, None]
        self.weights *= active
        if self.bones and np.any(self.bone_ids[active] >= len(self.bones)):
            raise FormatError('VVD referencia un hueso inexistente')
        self.bone_ids[~active] = 0
        last_error = None
        for suffix in ('.dx90.vtx', '.dx80.vtx', '.sw.vtx', '.vtx'):
            try:
                vtx = Reader(self.read(stem+suffix))
                break
            except (FileNotFoundError, KeyError) as error:
                last_error = error
        else:
            raise FileNotFoundError(f'No se encontró VTX para {stem}') from last_error
        if vtx.i(0) != 7 or vtx.i(16) != self.checksum:
            raise FormatError('VTX incompatible o checksum diferente al MDL')
        nbp, bpoff = vtx.unpack('ii', 28)
        mdl_nbp, mdl_bpoff = self.r.unpack('ii', 232)
        if nbp != mdl_nbp:
            raise FormatError('VTX bodyparts no coinciden con MDL')
        triangles, materials = [], []
        sg_stride, strip_stride = (33, 35) if self.version == 49 else (25, 27)
        for bi in vtx.table(bpoff, nbp, 8, 4096):
            bp = bpoff+bi*8
            nmodels, modeloff = vtx.unpack('ii', bp)
            mbp = mdl_bpoff+bi*16
            mnmodels, mmodeloff = self.r.i(mbp+4), self.r.i(mbp+12)
            if nmodels != mnmodels:
                raise FormatError('VTX models no coinciden con MDL')
            if nmodels > 1:
                self.warnings.append(f'Bodygroup {bi}: preview de primera variante')
            for mi in range(min(nmodels, 1)):
                model = bp+modeloff+mi*8
                lods, lodoff = vtx.unpack('ii', model)
                if not lods:
                    continue
                lod = model+lodoff
                nmeshes, meshoff = vtx.unpack('ii', lod)
                mm = mbp+mmodeloff+mi*148
                mnmeshes, mmeshoff = self.r.unpack('ii', mm+72)
                model_vertex_offset = self.r.i(mm+84)//48
                if nmeshes != mnmeshes:
                    raise FormatError('VTX meshes no coinciden con MDL')
                for mei in vtx.table(lod+meshoff, nmeshes, 9, 4096):
                    mesh = lod+meshoff+mei*9
                    ngroups, groupoff = vtx.unpack('ii', mesh)
                    mmsh = mm+mmeshoff+mei*116
                    material = self.r.i(mmsh)
                    base = model_vertex_offset+self.r.i(mmsh+12)
                    for gi in vtx.table(mesh+groupoff, ngroups, sg_stride, 65536):
                        sg = mesh+groupoff+gi*sg_stride
                        nv, vo, ni, io, ns, so = vtx.unpack('6i', sg)
                        vtx.table(sg+vo, nv, 9)
                        vtx.table(sg+io, ni, 2, 5000000)
                        original = np.frombuffer(vtx.data, dtype=np.dtype({'names':['id'], 'formats':['<u2'], 'offsets':[4], 'itemsize':9}), count=nv, offset=sg+vo)['id'].astype(int)+base
                        indices = np.frombuffer(vtx.data, dtype='<u2', count=ni, offset=sg+io)
                        if len(indices) and int(indices.max()) >= nv:
                            raise FormatError('Índice VTX fuera del stripgroup')
                        for si in vtx.table(sg+so, ns, strip_stride, 65536):
                            strip = sg+so+si*strip_stride
                            size, off = vtx.unpack('ii', strip)
                            flags = vtx.unpack('B', strip+18)[0]
                            if size < 0 or off < 0 or off+size > ni:
                                raise FormatError('Strip VTX fuera del buffer')
                            ids = original[indices[off:off+size]]
                            if flags & 1:
                                if size % 3:
                                    raise FormatError('Trilist VTX incompleta')
                                tris = ids.reshape(-1, 3)
                            elif flags & 2:
                                tris = np.array([(ids[j], ids[j+1+(j%2)], ids[j+2-(j%2)]) for j in range(size-2)], dtype=int).reshape(-1,3)
                            else:
                                raise FormatError('Topología VTX desconocida')
                            tris = tris[(tris[:,0] != tris[:,1]) & (tris[:,1] != tris[:,2]) & (tris[:,0] != tris[:,2])]
                            triangles.extend(tris)
                            materials.extend([material]*len(tris))
        self.triangles = np.asarray(triangles, dtype=int).reshape(-1, 3)
        self.material_ids = np.asarray(materials, dtype=int)
        if len(self.triangles) and (self.triangles.min() < 0 or self.triangles.max() >= len(self.positions)):
            raise FormatError('Triángulo fuera del buffer VVD')

    def _globals(self, positions, quats):
        output = np.repeat(np.eye(4)[None], len(self.bones), axis=0)
        complete, visiting = set(), set()
        def visit(i):
            if i in complete:
                return
            if i in visiting:
                raise FormatError('Jerarquía de huesos cíclica')
            visiting.add(i)
            output[i,:3,:3] = quaternion_matrix(quats[i])
            output[i,:3,3] = positions[i]
            parent = self.bones[i].parent
            if parent >= 0:
                visit(parent)
                output[i] = output[parent] @ output[i]
            visiting.remove(i)
            complete.add(i)
        for i in range(len(self.bones)):
            visit(i)
        return output

    def bind_pose(self):
        return self._globals([b.pos for b in self.bones], [b.quat for b in self.bones])

    def _anim_data(self, anim, frame):
        block, index = anim['block'], anim['index']
        if anim['sectionframes']:
            size = anim['sectionframes']
            # Last frame is stored in its own section for long animations.
            section = frame//size
            if anim['frames'] > size and frame == anim['frames']-1:
                section, frame = anim['frames']//size+1, 0
            else:
                frame %= size
            block, index = self.r.unpack('ii', anim['offset']+anim['sections']+section*8)
        if anim['flags'] & 0x20 or (index == 0 and block == 0):
            return self.r, None, frame
        if block == 0:
            return self.r, anim['offset']+index, frame
        if block < 0:
            raise FormatError('Bloque de animación inválido')
        count, offset = self.r.unpack('ii', 352)
        if block >= count:
            raise FormatError('Bloque ANI fuera del índice')
        start, end = self.r.unpack('ii', offset+block*8)
        if self._ani is None:
            name = self.r.string(self.r.i(348)).replace('\\', '/').lower()
            if not name.startswith('models/'):
                name = str(PurePosixPath(self.path).parent / name)
            self._ani = Reader(self.read(name))
        if start < 0 or end < start or end > len(self._ani.data):
            raise FormatError('Bloque ANI truncado')
        return self._ani, start+index, frame

    def _local_pose(self, anim_index, frame):
        anim = self.animations[anim_index]
        if anim['flags'] & 4:
            positions = np.zeros((len(self.bones), 3))
            quats = np.tile((0., 0., 0., 1.), (len(self.bones), 1))
        else:
            positions = np.array([b.pos for b in self.bones])
            quats = np.array([b.quat for b in self.bones])
        frame = int(max(0, min(frame, anim['frames']-1)))
        data, offset, frame = self._anim_data(anim, frame)
        if anim['flags'] & 0x40 and offset is not None:
            self._frame_pose(data, offset, frame, positions, quats)
            offset = None
        seen = set()
        while offset is not None:
            if offset in seen or len(seen) > len(self.bones):
                raise FormatError('Cadena de animación cíclica')
            seen.add(offset)
            bone, flags, nextoff = data.unpack('BBh', offset)
            if bone >= len(self.bones):
                raise FormatError('Animación referencia hueso inexistente')
            b = self.bones[bone]
            delta = bool(flags & 16)
            p = offset+4
            if flags & 2:
                quats[bone] = quat48(data, p)
            elif flags & 32:
                quats[bone] = quat64(data, p)
            elif flags & 8:
                values = data.unpack('3h', p)
                rotation = np.zeros(3) if delta else b.rot.copy()
                for axis, rel in enumerate(values):
                    if rel > 0:
                        rotation[axis] += rle_value(data, p+rel, frame)*b.rotscale[axis]
                quats[bone] = euler_quaternion(rotation)
            elif delta:
                quats[bone] = (0, 0, 0, 1)
            if flags & 1:
                raw = p+(6 if flags & 2 else 0)+(8 if flags & 32 else 0)
                positions[bone] = data.unpack('3e', raw)
            elif flags & 4:
                ptr = p+(6 if flags & 8 else 0)
                values = data.unpack('3h', ptr)
                positions[bone] = np.zeros(3) if delta else b.pos.copy()
                for axis, rel in enumerate(values):
                    if rel > 0:
                        positions[bone,axis] += rle_value(data, ptr+rel, frame)*b.posscale[axis]
            elif delta:
                positions[bone] = 0
            offset = offset+nextoff if nextoff > 0 else None
        if anim['flags'] & 4:
            # Delta sequences need reference-pose composition rather than an absolute pose.
            for i, bone in enumerate(self.bones):
                positions[i] += bone.pos
                x,y,z,w = bone.quat
                a,b,c,d = quats[i]
                quats[i] = (w*a+x*d+y*c-z*b, w*b-x*c+y*d+z*a, w*c+x*b-y*a+z*d, w*d-x*a-y*b-z*c)
        return positions, quats

    def _frame_pose(self, data, offset, frame, positions, quats):
        """Decode the v49 SDK frame-major format, including mixed constant tracks.

        Storage order per bone is rotation then position in each pool. Later
        CS:GO/Source Filmmaker extensions (Quat48S etc.) are rejected explicitly.
        """
        constant_offset, frame_offset, stride = data.unpack('3i', offset)
        data.table(offset+24, len(self.bones), 1, 256)
        if constant_offset < 0 or frame_offset < 0 or stride < 0:
            raise FormatError('Cabecera de animación por frames inválida')
        constants = offset+constant_offset
        cursor = offset+frame_offset+frame*stride
        frame_end = cursor+stride
        for i in range(len(self.bones)):
            flags = data.unpack('B', offset+24+i)[0]
            if flags & ~0x1f:
                raise FormatError('Extensión de frame anim no compatible (Quat48S/CS:GO)')
            if flags & 0x14 == 0x14 or flags & 0x03 and not constant_offset or flags & 0x1c and not frame_offset:
                raise FormatError('Flags/punteros de animación por frames inválidos')
            if flags & 2:
                quats[i] = quat48(data, constants)
                constants += 6
            if flags & 1:
                positions[i] = data.unpack('3e', constants)
                constants += 6
            if flags & 8:
                quats[i] = quat48(data, cursor)
                cursor += 6
            if flags & 4:
                positions[i] = data.unpack('3e', cursor)
                cursor += 6
            if flags & 16:
                positions[i] = data.unpack('3f', cursor)
                cursor += 12
        if cursor > frame_end:
            raise FormatError('Tracks exceden framelength de la animación')

    def pose(self, sequence_index=0, frame=0):
        if not self.sequences:
            return self.bind_pose()
        seq = self.sequences[sequence_index]
        if not seq.supported:
            return self.bind_pose()
        owner = seq.owner
        try:
            p0, q0 = owner._local_pose(seq.animation_index, math.floor(frame))
            alpha = float(frame)-math.floor(frame)
            if alpha > 1e-6:
                p1, q1 = owner._local_pose(seq.animation_index, math.floor(frame)+1)
                p0 = p0*(1-alpha)+p1*alpha
                dots = np.sum(q0*q1, axis=1)
                q1[dots < 0] *= -1
                q0 = q0*(1-alpha)+q1*alpha
                q0 /= np.maximum(np.linalg.norm(q0,axis=1)[:,None],1e-10)
            if owner is self:
                return self._globals(p0,q0)
            lookup = {bone.name:i for i,bone in enumerate(owner.bones)}
            p, q = [], []
            for bone in self.bones:
                i = lookup.get(bone.name)
                p.append(p0[i] if i is not None else bone.pos)
                q.append(q0[i] if i is not None else bone.quat)
            return self._globals(p,q)
        except (FileNotFoundError, KeyError, ValueError, struct.error) as error:
            seq.supported = False
            seq.warning = f'Animación no disponible: {error}'
            if seq.warning not in self.warnings:
                self.warnings.append(seq.warning)
            return self.bind_pose()

    def skinned_vertices(self, pose=None):
        if not self.bones or not len(self.positions):
            return self.positions.copy()
        pose = self.pose() if pose is None else pose
        transforms = pose @ np.array([bone.pose_to_bone for bone in self.bones])
        homogeneous = np.column_stack((self.positions, np.ones(len(self.positions))))
        mapped = np.einsum('nvij,nj->nvi', transforms[self.bone_ids], homogeneous)[:,:,:3]
        result = np.sum(mapped*self.weights[:,:,None], axis=1)
        zero = self.weights.sum(axis=1) < 1e-8
        result[zero] = self.positions[zero]
        return result


def load_model(read, path):
    return Model(read, path)
