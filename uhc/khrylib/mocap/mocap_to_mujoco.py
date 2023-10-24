import os
import sys

sys.path.append(os.getcwd())

from uhc.khrylib.utils import *
from uhc.khrylib.utils.transformation import quaternion_from_euler
from mujoco_py import load_model_from_path, MjSim, get_body_qposaddr
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
from uhc.khrylib.mocap.pose import load_amc_file, interpolated_traj
import pickle
import argparse
import glfw

parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--amc-id', type=str)
parser.add_argument('--seg-id', type=str)
parser.add_argument('--ext-id', type=str)
parser.add_argument('--version-id', type=str, default="2.0")
parser.add_argument('--mocap-fr', type=int, default=120)
parser.add_argument('--scale', type=float, default=0.45)
parser.add_argument('--dt', type=float, default=0.030)
parser.add_argument('--cyclic', action='store_true', default=False)
parser.add_argument('--cycle-start', type=int, default=5)
parser.add_argument('--cycle-end', type=int, default=60)
parser.add_argument('--offset-z', type=float, default=0.0)
args = parser.parse_args()

version2model = {'1.0': 'humanoid_pd_v2', '2.0': 'humanoid_pd_v4'}

select = args.seg_id is not None
model_file = 'assets/mujoco_models/%s.xml' % version2model[args.version_id]
model = load_model_from_path(model_file)
sim = MjSim(model)
viewer = MjViewer(sim)
body_qposaddr = get_body_qposaddr(model)

amc_sub = args.amc_id.split('_')[0]
amc_file = 'assets/amc/%s/%s.amc' % (amc_sub, args.amc_id)
scale = 1 / args.scale * 0.0254
poses, bone_addr = load_amc_file(amc_file, scale)
poses[:, bone_addr['lfoot'][0] + 2] = poses[:, bone_addr['lfoot'][0] + 2].clip(np.deg2rad(-10.0), np.deg2rad(10.0))
poses[:, bone_addr['rfoot'][0] + 2] = poses[:, bone_addr['lfoot'][0] + 2].clip(np.deg2rad(-10.0), np.deg2rad(10.0))

poses_samp = interpolated_traj(poses, args.dt, mocap_fr=args.mocap_fr)
expert_traj = []


def get_qpos(pose):
    qpos = np.zeros_like(sim.data.qpos)
    for bone_name, ind2 in body_qposaddr.items():
        ind1 = bone_addr[bone_name]
        if bone_name == 'root':
            trans = pose[ind1[0]:ind1[0] + 3].copy()
            trans[1], trans[2] = -trans[2], trans[1]
            angles = pose[ind1[0] + 3:ind1[1]].copy()
            quat = quaternion_from_euler(angles[0], angles[1], angles[2])
            quat[2], quat[3] = -quat[3], quat[2]
            qpos[ind2[0]:ind2[0] + 3] = trans
            qpos[ind2[0] + 3:ind2[1]] = quat
        else:
            qpos[ind2[0]:ind2[1]] = pose[ind1[0]:ind1[1]]
    return qpos


for i in range(poses_samp.shape[0]):
    cur_pose = poses_samp[i, :]
    cur_qpos = get_qpos(cur_pose)
    expert_traj.append(cur_qpos)

expert_traj = np.vstack(expert_traj)
expert_traj[:, 2] += args.offset_z

expert_qvel = []
for i in range(expert_traj.shape[0]):
    qpos = expert_traj[i]
    if i > 0:
        prev_qpos = expert_traj[i - 1]
        qvel = get_qvel_fd(prev_qpos, qpos, args.dt)
        expert_qvel.append(qvel)
expert_qvel.insert(0, expert_qvel[0].copy())
expert_qvel = np.vstack(expert_qvel)
"""render or select part of the clip"""
T = 10
fr = 0
paused = False
select_start = 0
select_end = expert_traj.shape[0]
g_offset = 0
stop = False


def key_callback(key, action, mods):
    global T, fr, paused, select_start, select_end, expert_traj, stop, g_offset

    if action != glfw.RELEASE:
        return False
    if key == glfw.KEY_D:
        T *= 1.5
        return True
    elif key == glfw.KEY_Q:
        select_start = fr
        return True
    elif key == glfw.KEY_W:
        select_end = fr + 1
        return True
    elif key == glfw.KEY_E:
        if select_end > select_start:
            expert_traj = expert_traj[select_start:select_end, :]
            g_offset += select_start
            select_end -= select_start
            select_start = 0
            fr = 0
        return True
    elif key == glfw.KEY_R:
        stop = True
        return True
    elif key == glfw.KEY_R:
        T = max(1, T / 1.5)
        return True
    elif key == glfw.KEY_RIGHT:
        fr = (fr + 1) % expert_traj.shape[0]
        return True
    elif key == glfw.KEY_LEFT:
        fr = (fr - 1) % expert_traj.shape[0]
        return True
    elif key == glfw.KEY_SPACE:
        paused = not paused
        return True
    return False


if select:
    viewer.custom_key_callback = key_callback

if args.render or select:
    t = 0
    while not stop:
        if t >= math.floor(T):
            fr = (fr + 1) % expert_traj.shape[0]
            t = 0
        sim.data.qpos[:] = expert_traj[fr]
        sim.forward()
        viewer.render()
        if not paused:
            t += 1

print('expert traj shape:', expert_traj.shape)
expert_meta = {'dt': args.dt, 'mocap_fr': args.mocap_fr, 'scale': args.scale, 'cyclic': args.cyclic, 'cycle_start': args.cycle_start, 'cycle_end': args.cycle_end, 'select': select, 'seg_id': args.seg_id, 'select_start': g_offset + select_start, 'select_end': g_offset + select_end}
print(expert_meta)
"""save the expert trajectory"""
out_id = args.amc_id
if args.seg_id is not None:
    out_id += '_%s' % args.seg_id
if args.ext_id is not None:
    out_id += '_%s' % args.ext_id
expert_traj_file = 'assets/expert_traj/%s/mocap_%s.p' % (args.version_id, out_id)
os.makedirs(os.path.dirname(expert_traj_file), exist_ok=True)
pickle.dump((expert_traj, expert_meta), open(expert_traj_file, 'wb'))
