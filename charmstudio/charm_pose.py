"""Explicit charm pose shared by preview and integrated export."""
import numpy as np

POSES = ('bind', 'sequence0')


def reference_pose(model, mode):
    if mode not in POSES:
        raise ValueError('Pose del charm no admitida')
    if mode == 'bind' or not model.sequences:
        return model.bind_pose()
    pose = model.pose(0, 0)
    if not model.sequences[0].supported:
        raise ValueError('No se puede leer la pose inicial del charm: '+model.sequences[0].warning)
    return pose


def reference_vertices(model, mode):
    return model.skinned_vertices(reference_pose(model, mode))


def skin_directions(model, pose, directions):
    """Rotate normals/tangents by the same weighted bind-to-pose matrices."""
    if not model.bones:
        return np.asarray(directions).copy()
    matrices = pose @ np.array([bone.pose_to_bone for bone in model.bones])
    rotated = np.einsum('nvij,nj->nvi', matrices[model.bone_ids,:,:][:,:,:3,:3], directions)
    output = (rotated*model.weights[:,:,None]).sum(axis=1)
    zero = model.weights.sum(axis=1)<1e-8
    output[zero] = directions[zero]
    return output/np.maximum(1e-12,np.linalg.norm(output,axis=1))[:,None]
