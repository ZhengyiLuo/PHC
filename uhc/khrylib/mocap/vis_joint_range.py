import argparse
import os
import sys
sys.path.append(os.getcwd())

from uhc.khrylib.utils import *
from mujoco_py import load_model_from_path, MjSim
from uhc.khrylib.rl.envs.common.mjviewer import MjViewer


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='assets/mujoco_models/human36m_orig.xml')
args = parser.parse_args()

model = load_model_from_path(args.model)
sim = MjSim(model)
viewer = MjViewer(sim)

jind = -1
jang = 30.0


def key_callback(key, action, mods):
    global jind, jang

    if action != glfw.RELEASE:
        return False
    elif key == glfw.KEY_LEFT:
        jind = max(jind - 1, -1)
        print('{} {} {}'.format(model.joint_names[jind + 1] if jind >= 0 else 'rest', jind, jang))
        return True
    elif key == glfw.KEY_RIGHT:
        jind = min(jind + 1, len(model.joint_names) - 2)
        print('{} {} {}'.format(model.joint_names[jind + 1] if jind >= 0 else 'rest', jind, jang))
        return True
    elif key == glfw.KEY_UP:
        jang += 5.0
        print('{} {} {}'.format(model.joint_names[jind + 1] if jind >= 0 else 'rest', jind, jang))
        return True
    elif key == glfw.KEY_DOWN:
        jang -= 5.0
        print('{} {} {}'.format(model.joint_names[jind + 1] if jind >= 0 else 'rest', jind, jang))
        return True
    return False


viewer._hide_overlay = True
viewer.custom_key_callback = key_callback
while True:
    sim.data.qpos[:] = 0.0
    sim.data.qpos[2] = 1.0
    if jind >= 0:
        sim.data.qpos[7 + jind] = math.radians(jang)
    sim.forward()
    viewer.render()

