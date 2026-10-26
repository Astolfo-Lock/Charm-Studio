import numpy as np


def transform(offset=(0, 0, 0), angles=(0, 0, 0), scale=1):
    """Source QAngle pitch/yaw/roll: Rz(yaw) Ry(pitch) Rx(roll)."""
    p, y, r = np.radians(angles)
    sp, cp, sy, cy, sr, cr = np.sin(p), np.cos(p), np.sin(y), np.cos(y), np.sin(r), np.cos(r)
    out = np.eye(4)
    out[:3, :3] = np.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
                              [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
                              [-sp, sr*cp, cr*cp]]) * scale
    out[:3, 3] = offset
    return out


def points(matrix, vertices):
    return vertices @ matrix[:3, :3].T + matrix[:3, 3]
