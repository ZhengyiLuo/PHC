from uhc.utils.math_utils import *

def get_body_qveladdr(model):
    body_qposaddr = dict()
    for i, body_name in enumerate(model.body_names):
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        end_joint = start_joint + model.body_jntnum[i]
        start_qposaddr = model.jnt_dofadr[start_joint]
        if end_joint < len(model.jnt_dofadr):
            end_qposaddr = model.jnt_dofadr[end_joint]
        else:
            end_qposaddr = model.nv
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr

def get_body_qposaddr(model):
    body_qposaddr = dict()
    for i, body_name in enumerate(model.body_names):
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        end_joint = start_joint + model.body_jntnum[i]
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr

def align_human_state(qpos, qvel, ref_qpos):
    qpos[:2] = ref_qpos[:2]
    hq = get_heading_q(ref_qpos[3:7])
    qpos[3:7] = quaternion_multiply(hq, qpos[3:7])
    qvel[:3] = quat_mul_vec(hq, qvel[:3])


def get_traj_pos(orig_traj):
    traj_pos = orig_traj[:, 2:].copy()
    for i in range(traj_pos.shape[0]):
        traj_pos[i, 1:5] = de_heading(traj_pos[i, 1:5])
    return traj_pos


def get_traj_vel(orig_traj, dt):
    traj_vel = []
    for i in range(orig_traj.shape[0] - 1):
        vel = get_qvel_fd(orig_traj[i, :], orig_traj[i + 1, :], dt, 'heading')
        traj_vel.append(vel)
    traj_vel.append(traj_vel[-1].copy())
    traj_vel = np.vstack(traj_vel)
    return traj_vel