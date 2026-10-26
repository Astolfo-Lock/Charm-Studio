"""Bounded Valve VPK v1/v2 reader and deterministic v1 writer."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import struct
import zlib


@dataclass(frozen=True)
class Entry:
    path: str
    crc: int
    archive: int
    offset: int
    size: int
    preload: bytes


class VPK:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = {}
        with self.path.open('rb') as f:
            header = f.read(12)
            if len(header) != 12:
                raise ValueError('VPK truncado')
            sig, version, tree_size = struct.unpack('<III', header)
            if sig != 0x55AA1234 or version not in (1, 2):
                raise ValueError('Se requiere VPK Source versión 1 o 2')
            self.data_start = (12 if version == 1 else 28) + tree_size
            if tree_size > 128 * 1024 * 1024:
                raise ValueError('Árbol VPK demasiado grande')
            f.seek(12 if version == 1 else 28)
            tree = f.read(tree_size)
        cursor = 0
        def string():
            nonlocal cursor
            end = tree.find(b'\0', cursor)
            if end < 0:
                raise ValueError('Árbol VPK truncado')
            s = tree[cursor:end].decode('utf-8', errors='replace')
            cursor = end + 1
            return s
        while ext := string():
            while folder := string():
                while name := string():
                    if cursor + 18 > len(tree):
                        raise ValueError('Entrada VPK truncada')
                    crc, pre, arc, off, size, term = struct.unpack_from('<IHHIIH', tree, cursor)
                    cursor += 18
                    if term != 0xffff or cursor + pre > len(tree):
                        raise ValueError('Entrada VPK inválida')
                    payload = tree[cursor:cursor+pre]
                    cursor += pre
                    rel = ('' if folder == ' ' else folder + '/') + name + ('' if ext == ' ' else '.' + ext)
                    if '..' in PurePosixPath(rel).parts or rel.startswith(('/', '\\')) or ':' in rel:
                        raise ValueError('Ruta VPK insegura')
                    self.entries[rel.lower()] = Entry(rel, crc, arc, off, size, payload)

    def read(self, name, verify=True):
        entry = self.entries[name.replace('\\', '/').lower()]
        if entry.size > 512 * 1024 * 1024:
            raise ValueError('Entrada demasiado grande')
        if entry.archive == 0x7fff:
            path, offset = self.path, self.data_start + entry.offset
        else:
            stem = self.path.stem
            if stem.endswith('_dir'):
                stem = stem[:-4]
            path, offset = self.path.with_name(f'{stem}_{entry.archive:03d}.vpk'), entry.offset
        with path.open('rb') as f:
            f.seek(offset)
            data = entry.preload + f.read(entry.size)
        if len(data) != len(entry.preload) + entry.size:
            raise ValueError(f'Entrada truncada: {name}')
        if verify and zlib.crc32(data) & 0xffffffff != entry.crc:
            raise ValueError(f'CRC incorrecto: {name}')
        return data


def write_vpk(path, files):
    groups = {}
    for name, data in sorted(files.items()):
        p = PurePosixPath(name.replace('\\', '/'))
        if p.is_absolute() or '..' in p.parts or ':' in name:
            raise ValueError('Ruta insegura')
        ext = p.suffix[1:] or ' '
        folder = str(p.parent) if str(p.parent) != '.' else ' '
        stem = p.name[:-len(p.suffix)] if p.suffix else p.name
        groups.setdefault(ext, {}).setdefault(folder, []).append((stem, data))
    tree, payload = bytearray(), bytearray()
    def s(value):
        tree.extend(value.encode('utf-8') + b'\0')
    for ext, folders in groups.items():
        s(ext)
        for folder, entries in folders.items():
            s(folder)
            for stem, data in entries:
                s(stem)
                tree.extend(struct.pack('<IHHIIH', zlib.crc32(data) & 0xffffffff, 0, 0x7fff, len(payload), len(data), 0xffff))
                payload.extend(data)
            s('')
        s('')
    s('')
    Path(path).write_bytes(struct.pack('<III', 0x55AA1234, 1, len(tree)) + tree + payload)
