import os
import sys
import time
import argparse
import pdb
import os.path as osp

sys.path.append(os.getcwd())

from phc.utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
import torch

import numpy as np
import math
from copy import deepcopy
from collections import defaultdict
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot
from easydict import EasyDict
from phc.utils.motion_lib_base import FixHeightMode
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused
    if chr(keycode) == "T":
        print("Next Motion")
        curr_start += num_motions
        motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
    elif chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    else:
        print("not mapped", chr(keycode))
    
    
        
if __name__ == "__main__":
    device = torch.device("cpu")
    motion_file = "sample_data/amass_isaac_standing_upright_slim.pkl"
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/30, False
    motion_lib_cfg = EasyDict({
                    "motion_file": motion_file,
                    "device": torch.device("cpu"),
                    "fix_height": FixHeightMode.full_fix,
                    "min_length": -1,
                    "max_length": -1,
                    "im_eval": False,
                    "multi_thread": False ,
                    "smpl_type": 'smpl',
                    "randomrize_heading": True,
                    "device": device,
                })
    
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": True,
        "remove_toe": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "model": "smpl",
        "big_ankle": True, 
        "freeze_hand": False,
        "box_body": True, 
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    smpl_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl",
    )
    
    gender_beta = np.zeros((17))
    smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
    test_good = f"/tmp/smpl/test_good.xml"
    smpl_robot.write_xml(test_good)
    smpl_robot.write_xml("test.xml")
    sk_tree = SkeletonTree.from_mjcf(test_good)
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    motion_lib.load_motions(skeleton_trees=[sk_tree] * num_motions, gender_betas=[torch.zeros(17)] * num_motions, limb_weights=[np.zeros(10)] * num_motions, random_sample=False, start_idx=curr_start)
    
    mj_model = mujoco.MjModel.from_xml_path(test_good)
    mj_data = mujoco.MjData(mj_model)

    
    # model = load_model_from_path(f"phc/data/assets/mjcf/amp_humanoid.xml")
    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(len(sk_tree._node_indices)):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.01, np.array([1, 0, 0, 1]))
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            step_start = time.time()
            motion_len = motion_lib.get_motion_length(motion_id).item()
            motion_time = time_step % motion_len
            motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(device), torch.tensor([motion_time]).to(device))

            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

            mj_data.qpos[:3] = root_pos[0].cpu().numpy()
            mj_data.qpos[3:7] = root_rot[0].cpu().numpy()[[3, 0, 1, 2]]
            mj_data.qpos[7:] = sRot.from_rotvec(dof_pos[0].cpu().numpy().reshape(-1, 3)).as_euler("XYZ").flatten()
            
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            for i in range(rb_pos.shape[1]):
                viewer.user_scn.geoms[i].pos = rb_pos[0, i]
                
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
