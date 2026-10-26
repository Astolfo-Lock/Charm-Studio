"""Mount independent addon folders, game content and split VPKs."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
from .vpk import VPK


class Mount:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.archive = VPK(self.path) if self.path.is_file() else None
        if self.archive:
            self.files = {key: e.path for key, e in self.archive.entries.items()}
        else:
            self.files = {}
            for root in ('models', 'materials', 'scripts'):
                folder = self.path / root
                if folder.is_dir():
                    for file in folder.rglob('*'):
                        if file.is_file():
                            rel = file.relative_to(self.path).as_posix()
                            self.files[rel.lower()] = rel

    @property
    def name(self):
        return self.path.name

    def read(self, name):
        key = normalize(name)
        if self.archive:
            return self.archive.read(key)
        rel = self.files[key]
        target = (self.path / rel).resolve()
        if not target.is_relative_to(self.path):
            raise ValueError('El recurso sale de la carpeta montada')
        if target.stat().st_size > 512 * 1024 * 1024:
            raise ValueError('Recurso demasiado grande')
        return target.read_bytes()


def normalize(name):
    name = str(name).replace('\\', '/').lower()
    if name.startswith('/') or '..' in PurePosixPath(name).parts or ':' in name:
        raise ValueError('Ruta de recurso inválida')
    return name


class Resources:
    def __init__(self):
        self.mounts = []

    def mount(self, path):
        path = Path(path).resolve()
        for existing in self.mounts:
            if existing.path == path:
                return existing
        result = Mount(path)
        self.mounts.append(result)
        return result

    def reader(self, preferred=None):
        # Model sidecars come from the selected addon first. Missing shared files
        # fall through to other addons/game packs, newest mount first.
        ordered = ([preferred] if preferred else []) + [m for m in reversed(self.mounts) if m is not preferred]
        def read(name):
            key = normalize(name)
            for mount in ordered:
                if key in mount.files:
                    return mount.read(key)
            raise FileNotFoundError(f'Recurso no encontrado: {name}. Monta el contenido base del juego si es compartido.')
        return read

    def mount_game(self, folder):
        folder = Path(folder)
        if (folder / 'left4dead2').is_dir():
            folder = folder / 'left4dead2'
        roots = [folder.parent / x for x in ('update', 'left4dead2_dlc3', 'left4dead2_dlc2', 'left4dead2_dlc1', 'left4dead2')]
        mounted = []
        # Higher priority DLC should be mounted last.
        for root in reversed(roots):
            if root.is_dir():
                for vpk in sorted(root.glob('*_dir.vpk')):
                    mounted.append(self.mount(vpk))
                if (root / 'models').exists() or (root / 'materials').exists():
                    mounted.append(self.mount(root))
        if not mounted:
            raise ValueError('No se encontraron pak*_dir.vpk ni carpetas models/materials en esa instalación.')
        return mounted

    def weapon_source(self, preferred, model_path):
        """Official update/DLC priority, preserving explicit Workshop sources."""
        content = preferred.path.parent if preferred.archive else preferred.path
        official = ('update', 'left4dead2_dlc3', 'left4dead2_dlc2', 'left4dead2_dlc1', 'left4dead2')
        if content.name.lower() not in official or not (content.parent/'left4dead2').is_dir():
            return preferred
        mounted = self.mount_game(content.parent)
        key = normalize(model_path)
        return next((mount for mount in reversed(mounted) if key in mount.files), preferred)
