import os
import sys
sys.path.append(os.getcwd())

from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer
import pickle
import argparse
import glfw
import math

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='human36m_v1')
parser.add_argument('--offset_z', type=float, default=0.0)
parser.add_argument('--start_take', type=str, default=None)
parser.add_argument('--dataset', type=str, default='h36m/data_qpos_h36m')
args = parser.parse_args()

model_file = f'assets/mujoco_models/{args.model_id}.xml'
model = load_model_from_path(model_file)
sim = MjSim(model)
viewer = MjViewer(sim)



def key_callback(key, action, mods):
    global T, fr, paused, stop, offset_z, take_ind, reverse

    if action != glfw.RELEASE:
        return False
    elif key == glfw.KEY_D:
        T *= 1.5
    elif key == glfw.KEY_F:
        T = max(1, T / 1.5)
    elif key == glfw.KEY_R:
        stop = True
    elif key == glfw.KEY_W:
        fr = 0
        update_mocap()
    elif key == glfw.KEY_S:
        reverse = not reverse
    elif key == glfw.KEY_C:
        take_ind = (take_ind + 1) % len(takes)
        load_take()
        update_mocap()
    elif key == glfw.KEY_Z:
        take_ind = (take_ind - 1) % len(takes)
        load_take()
        update_mocap()
    elif key == glfw.KEY_RIGHT:
        if fr < qpos_traj.shape[0] - 1:
            fr += 1
        update_mocap()
    elif key == glfw.KEY_LEFT:
        if fr > 0:
            fr -= 1
        update_mocap()
    elif key == glfw.KEY_UP:
        offset_z += 0.001
        update_mocap()
    elif key == glfw.KEY_DOWN:
        offset_z -= 0.001
        update_mocap()
    elif key == glfw.KEY_SPACE:
        paused = not paused
    else:
        return False
    return True


def update_mocap():
    print(f'{take[0]} {take[1]}: [{fr}, {qpos_traj.shape[0]}] dz: {offset_z:.3f}')
    print(qpos_traj.shape)
    sim.data.qpos[:] = qpos_traj[fr]
    sim.data.qpos[2] += offset_z
    sim.forward()


def load_take():
    global qpos_traj, fr, take
    take = takes[take_ind]
    fr = 0
    qpos_traj = data[take[0]][take[1]]


data = pickle.load(open(os.path.expanduser('data/{}.p').format(args.dataset), 'rb'))
takes = [(subject, action) for subject, s_data in data.items() for action in s_data.keys()]

qpos_traj = None
take = None
take_ind = 0 if args.start_take is None else takes.index(tuple(args.start_take.split(',')))
fr = 0
offset_z = args.offset_z
# load_take()

T = 10
paused = False
stop = False
reverse = False
glfw.set_window_size(viewer.window, 1000, 960)
glfw.set_window_pos(viewer.window, 400, 0)
viewer._hide_overlay = True
viewer.cam.distance = 10
viewer.cam.elevation = -20
viewer.cam.azimuth = 90
viewer.custom_key_callback = key_callback

load_take()
update_mocap()
t = 0
while not stop:
    if t >= math.floor(T):
        if not reverse and fr < qpos_traj.shape[0] - 1:
            fr += 1
            update_mocap()
        elif reverse and fr > 0:
            fr -= 1
            update_mocap()
        t = 0

    viewer.render()
    if not paused:
        t += 1



