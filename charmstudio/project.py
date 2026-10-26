"""Portable placement records and console commands (no console quoting needed)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path
import re
from .runtime_contract import RUNTIME_ENTRY, guarded_command, MAX_CONSOLE_BYTES, MAX_CFG_BYTES


def ascii_field(value):
    if not isinstance(value, str) or not value or len(value) > 160 or any(ord(c) < 32 or ord(c) > 126 for c in value):
        raise ValueError('Los campos deben tener entre 1 y 160 caracteres ASCII imprimibles')
    return value


def array(value):
    return '[' + ','.join(str(ord(c)) for c in ascii_field(value)) + ']'


@dataclass
class Placement:
    weapon: str
    weapon_sha256: str
    charm: str
    charm_sha256: str
    charm_id: str
    bone: str
    offset: list
    angles: list
    scale: float
    variant: str
    schema: int = 1
    charm_pose: str = 'bind'

    def validate(self):
        if self.schema != 1:
            raise ValueError('Versión de proyecto no admitida')
        if self.charm_pose not in ('bind','sequence0'):
            raise ValueError('Pose del charm no admitida')
        for field in (self.weapon, self.charm):
            ascii_field(field)
            if not field.startswith('models/') or not field.endswith('.mdl') or '..' in field or '\\' in field or ':' in field:
                raise ValueError('Ruta de modelo inválida')
        for field in (self.weapon_sha256, self.charm_sha256):
            if not re.fullmatch('[0-9a-f]{64}', field):
                raise ValueError('Huella SHA256 inválida')
        for field in (self.charm_id, self.bone, self.variant):
            ascii_field(field)
        for values in (self.offset, self.angles):
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                raise ValueError('Se requieren tres coordenadas')
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 10000 for v in values):
                raise ValueError('Coordenadas fuera de rango')
        if not isinstance(self.scale, (int, float)) or not math.isfinite(self.scale) or not 0 < self.scale <= 10:
            raise ValueError('La escala debe ser > 0 y <= 10')
        return self

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def load(cls, path):
        data = Path(path).read_bytes()
        if len(data) > 65536:
            raise ValueError('Proyecto demasiado grande')
        return cls(**json.loads(data)).validate()

    def commands(self):
        self.validate()
        def vector(values):
            return 'Vector(' + ','.join(format(x, '.6f') for x in values) + ')'
        lines = [
            f'script_execute {RUNTIME_ENTRY}',
            'script ::CSCfg <- {i=[],m=[],b=[],v=[],w=[],ok=false}',
        ]
        for key, value in (('i',self.charm_id),('m',self.charm),('b',self.bone),('v',self.variant),('w',self.weapon)):
            for start in range(0,len(value),24):
                lines.append(f'script CSCfg.{key}.extend({array(value[start:start+24])})')
        lines.extend([
            f'script CSCfg.p <- {vector(self.offset)}',
            f'script CSCfg.a <- {vector(self.angles)}',
            f'script CSCfg.s <- {self.scale:.6f}',
            'script CSCfg.api <- ("ChainsSystem" in getroottable())?ChainsSystem:{}',
            guarded_command('StudioRegister', 'CSCfg.i,CSCfg.m'),
            guarded_command('Studio', 'CSCfg.i,CSCfg.b,CSCfg.p,CSCfg.a,CSCfg.s,CSCfg.v,CSCfg.w'),
            'script delete ::CSCfg',
        ])
        if any(len(line.encode('ascii')) > MAX_CONSOLE_BYTES for line in lines):
            raise ValueError('El comando supera el límite conservador de la consola')
        result = '\n'.join(lines)
        if len(result.encode('ascii')) > MAX_CFG_BYTES:
            raise ValueError('El perfil supera el presupuesto del buffer de consola')
        return result


def digest(data):
    return hashlib.sha256(data).hexdigest()


def charm_id(path):
    return re.sub('[^a-z0-9_]', '_', Path(path).stem.lower())[:40] + '_' + digest(path.encode())[:6]
