import torch
import math
import numpy as np
from uhc.utils.transformation import (
    quaternion_matrix,
    quaternion_about_axis,
    quaternion_inverse,
    quaternion_multiply,
    rotation_from_quaternion,
    rotation_from_matrix,
)


def ewma(x, alpha=0.05):
    avg = x[0]
    for i in x[1:]:
        avg = alpha * i + (1 - alpha) * avg
    return avg


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_qvel_fd(cur_qpos, next_qpos, dt, transform=None):
    v = (next_qpos[:3] - cur_qpos[:3]) / dt
    qrel = quaternion_multiply(next_qpos[3:7], quaternion_inverse(cur_qpos[3:7]))
    # qrel /= np.linalg.norm(qrel)
    axis, angle = rotation_from_quaternion(qrel, True)

    if angle > np.pi:  # -180 < angle < 180
        angle -= 2 * np.pi  #
    elif angle < -np.pi:
        angle += 2 * np.pi

    rv = (axis * angle) / dt
    rv = transform_vec(rv, cur_qpos[3:7], "root")
    qvel = (next_qpos[7:] - cur_qpos[7:]) / dt
    qvel = np.concatenate((v, rv, qvel))
    if transform is not None:
        v = transform_vec(v, cur_qpos[3:7], transform)
        qvel[:3] = v
    return qvel


def get_angvel_fd(prev_bquat, cur_bquat, dt):
    q_diff = multi_quat_diff(cur_bquat, prev_bquat)
    n_joint = q_diff.shape[0] // 4
    body_angvel = np.zeros(n_joint * 3)
    for i in range(n_joint):
        body_angvel[3 * i : 3 * i + 3] = (
            rotation_from_quaternion(q_diff[4 * i : 4 * i + 4]) / dt
        )
    return body_angvel


def transform_vec(v, q, trans="root"):
    if trans == "root":
        rot = quaternion_matrix(q)[:3, :3]
    elif trans == "heading":
        hq = q.copy()
        hq[1] = 0.0
        hq[2] = 0.0
        hq /= np.linalg.norm(hq)
        rot = quaternion_matrix(hq)[:3, :3]
    else:
        assert False
    v = rot.T.dot(v[:, None]).ravel()
    return v


def get_heading_q(q):
    hq = q.copy()
    hq[1] = 0.0
    hq[2] = 0.0
    hq /= np.linalg.norm(hq)
    return hq


def get_heading(q):
    hq = q.copy()
    hq[1] = 0
    hq[2] = 0
    if hq[3] < 0:
        hq *= -1
    hq /= np.linalg.norm(hq)
    return 2 * math.acos(hq[0])


def de_heading(q):
    return quaternion_multiply(quaternion_inverse(get_heading_q(q)), q)


def multi_quat_diff(nq1, nq0):
    """return the relative quaternions q1-q0 of N joints"""

    nq_diff = np.zeros_like(nq0)
    for i in range(nq1.shape[0] // 4):
        ind = slice(4 * i, 4 * i + 4)
        q1 = nq1[ind]
        q0 = nq0[ind]
        nq_diff[ind] = quaternion_multiply(q1, quaternion_inverse(q0))
    return nq_diff


def multi_quat_norm(nq):
    """return the scalar rotation of a N joints"""

    nq_norm = np.arccos(np.clip(nq[::4], -1.0, 1.0))
    return nq_norm


def multi_quat_norm_v2(nq):

    _diff = []
    for i in range(nq.shape[0] // 4):
        q = nq[4 * i : 4 * (i + 1)]
        d = np.array([abs(q[0]) - 1.0, q[1], q[2], q[3]])
        _diff.append(np.linalg.norm(d))
    return np.array(_diff)


def quat_mul_vec(q, v):
    return quaternion_matrix(q)[:3, :3].dot(v[:, None]).ravel()


def quat_to_bullet(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_from_bullet(q):
    return np.array([q[3], q[0], q[1], q[2]])


def quat_from_expmap(e):
    angle = np.linalg.norm(e)
    if angle < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        angle = 0.0
    else:
        axis = e / angle
    return quaternion_about_axis(angle, axis)


def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q - 1] - quat[q], axis=0) > np.linalg.norm(
            quat[q - 1] + quat[q], axis=0
        ):
            quat[q] = -quat[q]
    return quat
