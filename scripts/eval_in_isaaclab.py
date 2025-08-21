# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""

import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
import argparse
from isaaclab.app import AppLauncher
import sys

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--motion_file", type=str, default="sample_data/amass_isaac_standing_upright_slim.pkl", help="Path to motion file to load.")
parser.add_argument("--policy_path", type=str, default="output/HumanoidIm/phc_3/Humanoid.pth", help="Path to policy checkpoint file.")
parser.add_argument("--action_offset_file", type=str, default="phc/data/action_offset_smpl.pkl", help="Path to action offset file.")
parser.add_argument("--humanoid_type", type=str, default="smpl", choices=["smpl", "smplx"], help="Type of humanoid model to use.")
parser.add_argument("--num_motions", type=int, default=10, help="Number of motions to load from motion library.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard
import isaaclab.utils.math as lab_math_utils
import carb
import imageio
from carb.input import KeyboardEventType



##
# Pre-defined configs
##
# from isaaclab_assets import CARTPOLE_CFG  # isort:skip
from phc.assets.smpl_config import SMPL_Upright_CFG, SMPL_CFG, SMPLX_Upright_CFG, SMPLX_CFG

from smpl_sim.utils.rotation_conversions import xyzw_to_wxyz, wxyz_to_xyzw
from collections.abc import Sequence
from phc.utils.flags import flags
from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from phc.learning.network_loader import load_z_encoder, load_z_decoder, load_pnn, load_mcp_mlp
from phc.utils.isaacgym_humanoid_funcs import compute_humanoid_observations_smpl_max, compute_imitation_observations_v6 
from rl_games.algos_torch import torch_ext
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from rl_games.algos_torch.players import rescale_actions
import torch
import joblib
from easydict import EasyDict
import numpy as np
import copy
from scipy.spatial.transform import Rotation as sRot
import time



flags.test=False

@configclass
class SMPLSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

@configclass 
class SMPLEnvCfg(DirectRLEnvCfg):
    num_actions = 69
    num_observations = 1
    num_states = 1
    
    decimation = 2
    episode_length_s = 15.0
    
    sim = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1 / 60,
        # decimation will be set in the task config
        # up axis will always be Z in isaac sim
        # use_gpu_pipeline is deduced from the device
        gravity=(0.0, 0.0, -9.81),
        physx = PhysxCfg(
            # num_threads is no longer needed
            solver_type=1,
            # use_gpu is deduced from the device
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            # moved to actor config
            # moved to actor config
            bounce_threshold_velocity=0.2,
            # moved to actor config
            # default_buffer_size_multiplier is no longer needed
            gpu_max_rigid_contact_count=2**23,
            # num_subscenes is no longer needed
            # contact_collection is no longer needed
        )
    )
    

    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))
    
    smpl_robot: ArticulationCfg = SMPL_Upright_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # smpl_robot: ArticulationCfg = SMPLX_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # scene
    scene: InteractiveSceneCfg = SMPLSceneCfg(num_envs=args_cli.num_envs, env_spacing=20.0, replicate_physics=True)
    
    observation_space = None
    action_space = None
    motion_file = args_cli.motion_file
    

class SMPLEnv(DirectRLEnv):
    cfg: SMPLEnvCfg
    
    def __init__(self, cfg: SMPLEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        # self.humanoid_type = "smplx"   
        self.humanoid_type = "smpl"   
        super().__init__(cfg, render_mode, **kwargs)
        
        SMPL_NAMES = SMPLH_MUJOCO_NAMES if self.humanoid_type == "smplx" else SMPL_MUJOCO_NAMES
        self.gym_joint_names = gym_joint_names = [f"{j}_{axis}" for j in SMPL_NAMES[1:] for axis in ["x", "y", "z"]]
        self.sim_joint_names = sim_joint_names = self.robot.data.joint_names
        self.sim_body_names = sim_body_names = self.robot.data.body_names
        
        self.sim_to_gym_body = [sim_body_names.index(n) for n in SMPL_NAMES]
        self.sim_to_gym_dof = [sim_joint_names.index(n) for n in gym_joint_names]
        self.gym_to_sim_dof = [gym_joint_names.index(n) for n in sim_joint_names]
        self.gym_to_sim_body = [SMPL_NAMES.index(n) for n in sim_body_names]
        
        self._load_motion(cfg.motion_file)
        
        keyboard_interface = Se2Keyboard()
        keyboard_interface.add_callback("R", self.reset)
        
        self.rigid_body_pos = self.robot.data.body_pos_w
        
    
    def close(self):
        super().close()
        
    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states
        
        
    def _setup_scene(self):
        self.robot = robot = Articulation(self.cfg.smpl_robot)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[])

        # add articultion and sensors to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        
    def _load_motion(self, motion_file):
        
        time_steps = 1
        if self.humanoid_type == "smplx":
            self._has_upright_start = False
        else:
            self._has_upright_start = True
        
        motion_lib_cfg = EasyDict({
                        "has_upright_start": self._has_upright_start,
                        "motion_file": motion_file,
                        "fix_height": FixHeightMode.full_fix,
                        "min_length": -1,
                        "max_length": 3000,
                        "im_eval": flags.im_eval,
                        "multi_thread": False ,
                        "smpl_type": self.humanoid_type,
                        "randomrize_heading": True,
                        "device": self.device,
                        "real_traj": False,
                        "simulator": "isaac_sim",
                        "gym_to_sim_dict": {
                            "gym_to_sim_dof": self.gym_to_sim_dof,
                            "gym_to_sim_body": self.gym_to_sim_body,
                            "sim_to_gym_dof": self.sim_to_gym_dof,
                            "sim_to_gym_body": self.sim_to_gym_body,
                        },
                        "test": True, 
                        "im_eval": False,
                    })
        robot_cfg = {
                "mesh": False,
                "rel_joint_lm": False,
                "upright_start": self._has_upright_start,
                "remove_toe": False,
                "real_weight_porpotion_capsules": True,
                "real_weight_porpotion_boxes": True,
                "model": self.humanoid_type,
                "big_ankle": True, 
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
            
        if self.humanoid_type == "smplx":
            gender_beta = np.zeros((17))
        else:
            gender_beta = np.zeros((11))
            
        smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
        test_good = f"/tmp/smpl/test_good.xml"
        smpl_robot.write_xml(test_good)
        sk_tree = SkeletonTree.from_mjcf(test_good)
        num_motions = 10
        skeleton_trees = [sk_tree] * num_motions
        start_idx = 0
        
        motion_lib = MotionLibSMPL(motion_lib_cfg)
        motion_lib.load_motions(skeleton_trees=skeleton_trees, 
                                gender_betas=[torch.from_numpy(gender_beta)] * num_motions,
                                limb_weights=[torch.from_numpy(gender_beta)] * num_motions, # not used.
                                random_sample=False,
                                start_idx = start_idx)
        self._motion_lib = motion_lib
        self._motion_id, self._motion_time = torch.arange(self.num_envs).to(self.device).long(), torch.zeros(self.num_envs).to(self.device).float()
        
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions
        
        pass
    
    def _post_physics_step(self) -> None:
        
        
        pass
        
    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions, joint_ids=None)
        
    def _get_observations(self) -> dict:
        
        motion_time = (self.episode_length_buf + 1) * self.step_dt + self._motion_time
        
        motion_res = self._motion_lib.get_motion_state(self._motion_id, motion_time)
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                        motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                        motion_res["motion_bodies"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        
        body_pos = self.robot.data.body_pos_w[:, self.sim_to_gym_body ]
        body_rot = wxyz_to_xyzw(self.robot.data.body_quat_w[:, self.sim_to_gym_body])
        body_vel = self.robot.data.body_lin_vel_w[:, self.sim_to_gym_body]
        body_ang_vel = self.robot.data.body_ang_vel_w[:, self.sim_to_gym_body]
        
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]
        body_pos_subset = body_pos
        body_rot_subset = body_rot
        body_vel_subset = body_vel
        body_ang_vel_subset = body_ang_vel
        
        ref_rb_pos_subset = ref_rb_pos
        ref_rb_rot_subset = ref_rb_rot
        ref_body_vel_subset = ref_body_vel
        ref_body_ang_vel_subset = ref_body_ang_vel
        
        # Data replay
        # ref_joint_pos = ref_dof_pos[:, self.gym_to_sim_dof]
        # ref_joint_vel = ref_dof_vel[:, self.gym_to_sim_dof]
        
        # ref_root_state = torch.cat([ref_root_pos, xyzw_to_wxyz(ref_root_rot), ref_root_vel, ref_root_ang_vel], dim = -1)
        # self.robot.write_root_state_to_sim(ref_root_state, None)
        # self.robot.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, None, None)
        
        self_obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, ref_smpl_params, torch.tensor(0), True, True, self._has_upright_start, False, False)
        task_obs = compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, 1, self._has_upright_start)
        
        return {
            "self_obs": self_obs,
            "task_obs": task_obs,
        }
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs), torch.zeros(self.num_envs)
    
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._motion_time[env_ids] = 0
        
        motion_res = self._motion_lib.get_motion_state(self._motion_id, self._motion_time)
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                        motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                        motion_res["motion_bodies"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        init_joint_pos = ref_dof_pos[:, self.gym_to_sim_dof]
        init_joint_vel = ref_dof_vel[:, self.gym_to_sim_dof]
        
        
        init_root_state = torch.cat([ref_root_pos, xyzw_to_wxyz(ref_root_rot), ref_root_vel, ref_root_ang_vel], dim = -1)
        self.robot.write_root_state_to_sim(init_root_state, env_ids)
        self.robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel, None, env_ids)
    
    
def main():
    """Main function."""
    env_cfg = SMPLEnvCfg()
    env = SMPLEnv(env_cfg)
    
    
    device = env.device
    
    # Use configurable policy path or fall back to defaults
    policy_path = args_cli.policy_path
    
    
    check_points = [torch_ext.load_checkpoint(policy_path)]
    pnn = load_pnn(check_points[0], num_prim = 3, has_lateral = False, activation = "silu", device = device)
    running_mean, running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']
    
    action_offset = joblib.load(args_cli.action_offset_file)
        
    pd_action_offset = action_offset[0]
    pd_action_scale = action_offset[1]
    
    time = 0 
    obs_dict, extras = env.reset()
    while True:
        self_obs, task_obs = obs_dict["self_obs"], obs_dict["task_obs"]
        full_obs = torch.cat([self_obs, task_obs], dim = -1)
        full_obs = ((full_obs - running_mean.float()) / torch.sqrt(running_var.float() + 1e-05))
        full_obs = torch.clamp(full_obs, min=-5.0, max=5.0)
        
        
        with torch.no_grad():
            actions, _ = pnn(full_obs, idx=0)
            actions = rescale_actions(-1, 1, torch.clamp(actions, -1, 1))
            actions = actions * pd_action_scale + pd_action_offset
            actions = actions[:, env.gym_to_sim_dof]
        
        obs_dict, _, _, _, _ = env.step(actions)
    


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    
    