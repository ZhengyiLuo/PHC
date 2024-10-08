

import os.path as osp
from typing import OrderedDict
import torch
import numpy as np
from phc.utils.torch_utils import quat_to_tan_norm
import phc.env.tasks.humanoid_im as humanoid_im

from phc.env.tasks.humanoid_amp import HumanoidAMP, remove_base_rot
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils.motion_lib_real import MotionLibReal

from phc.utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from phc.utils.flags import flags
import joblib
import gc
from collections import defaultdict

from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import open3d as o3d
from datetime import datetime
import imageio
from collections import deque
from tqdm import tqdm
import copy
from phc.utils.lpf import ActionFilterButter, ActionFilterExp, ActionFilterButterTorch
from phc.utils.draw_utils import agt_color, get_color_gradient

class HumanoidTeleop(humanoid_im.HumanoidIm):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
        self._prepare_reward_function()
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self._dof_vel)
        self.last_root_vel = torch.zeros_like(self._humanoid_root_states[:, 7:13])
        self._recovery_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._recovery_steps = 60
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.feet_indices = self._contact_body_ids
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        if self.cfg.control.action_filter:
            self.action_filter = ActionFilterButterTorch(lowcut=np.zeros(self.num_envs*self.num_actions),
                                                        highcut=np.ones(self.num_envs*self.num_actions) * self.cfg.control.action_cutfreq, 
                                                        sampling_rate=1./self.dt, num_joints=self.num_envs * self.num_actions, 
                                                        device=self.device)

        if self.cfg.domain_rand.randomize_ctrl_delay:
            
            self.action_queue = torch.zeros(self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0], 
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)
            
        self.common_step_counter = 0
        
        self.self_noise_vec, self.task_noise_vec = self._get_noise_scale_vec()
        self.noise_scale_vec = torch.cat([self.self_noise_vec, self.task_noise_vec], dim=-1)
    
    def _physics_step(self):
        actions = self.actions
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = actions.clone()
            actions = self.action_queue[torch.arange(self.num_envs), self.action_delay].clone()
            
            
        if self.cfg.control.action_filter:
            actions = self.action_filter.filter(actions.reshape(self.num_envs * self.num_actions)).reshape(self.num_envs, self.num_actions)
        else:
            actions = actions.clone()
        
        
        for i in range(self.control_freq_inv):
            self.render(i = i) 
            if not self.paused and self.enable_viewer_sync:
                if self.humanoid_type in ['h1', 'g1']:
                    self.torques = self._compute_torques(actions)
                    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                    self.gym.simulate(self.sim)
                    if self.device == 'cpu':
                        self.gym.fetch_results(self.sim, True)
                    self.gym.refresh_dof_state_tensor(self.sim)
                else:
                    self.gym.simulate(self.sim)
        return
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self._humanoid_root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._humanoid_root_states))
        self._recovery_counter[:] = 60 # 60 steps for the robot to stabilize
        
        
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        
        self._recovery_counter[env_ids] = 0
        
        if self.cfg.control.action_filter:
            # # older version
            # filter_action_ids = np.concatenate([np.arange(self.num_actions) + env_id.cpu().numpy() * self.num_actions for env_id in env_ids])
            # self.action_filter.reset_by_ids(filter_action_ids)
            filter_action_ids_torch = torch.concat([torch.arange(self.num_actions,dtype=torch.long, device=self.device) + env_id * self.num_actions for env_id in env_ids])
            self.action_filter.reset_hist(filter_action_ids_torch)
            
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],  self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)
    
    def _update_recovery_count(self):
        self._recovery_counter -= 1
        self._recovery_counter = torch.clamp_min(self._recovery_counter, 0)
        return
    
    def _compute_observations(self, env_ids=None):
        # env_ids is used for resetting
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        self_obs = self._compute_humanoid_obs(env_ids)
        self.self_obs_buf[env_ids] = self_obs

        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([self_obs, task_obs], dim=-1)
        else:
            obs = self_obs
            
        if self.add_noise:
            obs += (2 * torch.rand_like(obs) - 1) * self.noise_scale_vec
        
        if self.obs_v == 4:
            # Double sub will return a copy.
            B, N = obs.shape
            sums = self.obs_buf[env_ids, 0:self.past_track_steps].abs().sum(dim=1)
            zeros = sums == 0
            nonzero = ~zeros
            obs_slice = self.obs_buf[env_ids]
            obs_slice[zeros] = torch.tile(obs[zeros], (1, self.past_track_steps))
            obs_slice[nonzero] = torch.cat([obs_slice[nonzero, N:], obs[nonzero]], dim=-1)
            self.obs_buf[env_ids] = obs_slice
        else:
            self.obs_buf[env_ids] = obs
        return obs
    
    def post_physics_step(self):
        super().post_physics_step()
        self._update_recovery_count()
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self._dof_vel[:]
        self.last_root_vel[:] = self._humanoid_root_states[:, 7:13]
        self.common_step_counter += 1
        
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
            
    def _compute_reset(self):
        super()._compute_reset()

        is_recovery = self._recovery_counter > 0
        self.reset_buf[is_recovery] = 0
        self._terminate_buf[is_recovery] = 0
        self.progress_buf[is_recovery] -= 1  # ZL: do not advance progress buffer for these.
        return
    
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = copy.deepcopy(self.reward_specs)
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] = scale * self.dt
                
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        
        for name, scale in self.reward_scales.items():
            if name.startswith("r_"):
                if name=="termination":
                    continue
                self.reward_names.append(name)
                name = '_reward_' + name[2:]
                self.reward_functions.append(getattr(self, name))
    
    
    def _compute_reward(self, actions):
        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel

        motion_times = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset  # reward is computed after phsycis step, and progress_buf is already updated for next time step.

        motion_res = self._get_state_from_motionlib_cache(self._sampled_motion_ids, motion_times, self._global_offset) 

        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        root_pos = body_pos[..., 0, :]
        root_rot = body_rot[..., 0, :]

         

        if self.humanoid_type in ['h1', 'g1',]:
            extend_curr_pos = torch_utils.my_quat_rotate(body_rot[:, self.extend_body_parent_ids].reshape(-1, 4), self.extend_body_pos_in_parent.reshape(-1, 3)).view(self.num_envs, -1, 3) + body_pos[:, self.extend_body_parent_ids]
            body_pos_extend = torch.cat([body_pos, extend_curr_pos], dim=1)
            body_rot_extend = torch.cat([body_rot, body_rot[:, self.extend_body_parent_ids]], dim=1)
            ref_rb_pos_extend = torch.cat([ref_rb_pos, motion_res["rg_pos_t"][:, self.num_bodies:]], dim = 1)
            ref_rb_rot_extend = torch.cat([ref_rb_rot, motion_res["rg_rot_t"][:, self.num_bodies:]], dim = 1)
            
            self.rew_buf[:], self.reward_raw = humanoid_im.compute_imitation_reward(root_pos, root_rot, body_pos_extend, body_rot_extend, body_vel, body_ang_vel, ref_rb_pos_extend, ref_rb_rot_extend, ref_body_vel, ref_body_ang_vel, self.reward_specs)
        else:
            self.rew_buf[:], self.reward_raw = humanoid_im.compute_imitation_reward(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel, self.reward_specs)
    
        # print(self.dof_force_tensor.abs().max())
        if self.power_reward:
            power = torch.abs(torch.multiply(self.dof_force_tensor, self._dof_vel)).sum(dim=-1) 
            # power_reward = -0.00005 * (power ** 2)
            power_reward = -self.power_coefficient * power
            power_reward[self.progress_buf <= 3] = 0 # First 3 frame power reward should not be counted. since they could be dropped.

            self.rew_buf[:] += power_reward
            self.reward_raw = torch.cat([self.reward_raw, power_reward[:, None]], dim=-1)
            
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.reward_raw = torch.cat([self.reward_raw, rew[:, None]], dim=-1)
        
        return
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self._dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self._dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*1).clip(min=0.), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self._dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self._dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_slippage(self):
        foot_vel = self._rigid_body_vel[:, self.feet_indices]
        return torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(self._contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self._contact_forces[:, self.feet_indices, :], dim=-1) -  self.reward_specs.max_contact_force).clip(min=0.), dim=1)

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self._contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self._contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_feet_air_time_teleop(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self._contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.ref_body_vel[:, 0, :2], dim=1) > 0.1 #no reward for low ref motion velocity (root xy velocity)
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_feet_ori(self):
        left_quat = self._rigid_body_rot[:, self.feet_indices[0]]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_quat = self._rigid_body_rot[:, self.feet_indices[1]]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _init_domain_params(self):
        # init params for domain randomization
        # init 0 for values
        # init 1 for scales
        self._base_com_bias = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._ground_friction_values = torch.zeros(self.num_envs, self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)        
        self._link_mass_scale = torch.ones(self.num_envs, len(self.cfg.domain_rand.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rfi_lim_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
     
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # sum_mass = 0
        # print(env_id)
        # for i in range(len(props)):
        #     print(f"Mass of body {i}: {props[i].mass} (before randomization)")
        #     sum_mass += props[i].mass
        
        # print(f"Total mass {sum_mass} (before randomization)")
        # print()
        
        # randomize base com
        self._body_list = self._body_names
        if self.cfg.domain_rand.randomize_base_com:
            torso_index = self._body_list.index("torso_link")
            assert torso_index != -1

            com_x_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.x[0], self.cfg.domain_rand.base_com_range.x[1])
            com_y_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.y[0], self.cfg.domain_rand.base_com_range.y[1])
            com_z_bias = np.random.uniform(self.cfg.domain_rand.base_com_range.z[0], self.cfg.domain_rand.base_com_range.z[1])

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            props[torso_index].com.x += com_x_bias
            props[torso_index].com.y += com_y_bias
            props[torso_index].com.z += com_z_bias

        # randomize link mass
        if self.cfg.domain_rand.randomize_link_mass:
            for i, body_name in enumerate(self.cfg.domain_rand.randomize_link_body_names):
                body_index = self._body_list.index(body_name)
                assert body_index != -1

                mass_scale = np.random.uniform(self.cfg.domain_rand.link_mass_range[0], self.cfg.domain_rand.link_mass_range[1])
                props[body_index].mass *= mass_scale

                self._link_mass_scale[env_id, i] *= mass_scale

        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            raise Exception("index 0 is for world, 13 is for torso!")
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        sum_mass = 0
        # print(env_id)
        # for i in range(len(props)):
        #     print(f"Mass of body {i}: {props[i].mass} (after randomization)")
        #     sum_mass += props[i].mass
        
        # print(f"Total mass {sum_mass} (afters randomization)")
        # print()

        return props
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
                # import pdb; pdb.set_trace()
                self._ground_friction_values[env_id, s] += self.friction_coeffs[env_id].squeeze()
        return props
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        self_obs_noise = torch.zeros(self.get_self_obs_size(), device=self.device)
        task_obs_noise = torch.zeros(self.get_task_obs_size(), device=self.device)
        self.add_noise = self.cfg.domain_rand.add_noise
        noise_scales = self.cfg.domain_rand.noise_scales
        noise_level = self.cfg.domain_rand.noise_level
         
        if self.self_obs_v == 4: 
            self_obs_noise[0                   : self.num_dof      ] = noise_scales.dof_pos * noise_level 
            # dof vel
            self_obs_noise[self.num_dof        : 2*self.num_dof    ] = noise_scales.dof_vel * noise_level 
            # base vel
            self_obs_noise[2*self.num_dof      : 2*self.num_dof + 3] = noise_scales.lin_vel * noise_level 
            # base ang vel
            self_obs_noise[2*self.num_dof + 3  : 2*self.num_dof + 6] = noise_scales.ang_vel * noise_level 
            # base gravity
            self_obs_noise[2*self.num_dof + 6  : 2*self.num_dof + 9] = noise_scales.gravity * noise_level
            # ref dof pos
            
        if self.obs_v == 13:
            task_obs_noise[:] = noise_scales.ref_body_pos * noise_level 
            
        return self_obs_noise, task_obs_noise