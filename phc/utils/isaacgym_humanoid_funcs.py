import torch
import joblib
import phc.utils.isaacgym_torch_utils as torch_utils
from phc.utils.isaacgym_torch_utils import *
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot

from tqdm import tqdm
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
from collections import defaultdict
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import gc

@torch.jit.script
def remove_base_rot(quat):
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat)) #SMPL
    shape = quat.shape[0]
    return quat_mul(quat, base_rot.repeat(shape, 1))



@torch.jit.script
def compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, smpl_params, limb_weight_params, local_root_obs, root_height_obs, upright, has_smpl_params, has_limb_weight_params):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, bool, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not upright:
        root_rot = remove_base_rot(root_rot)
    heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

    if (not root_height_obs):
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2])

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not (local_root_obs):
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot) # If not local root obs, you override it. 
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_list = []
    if root_height_obs:
        obs_list.append(root_h_obs)
    obs_list += [local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel]
    if has_smpl_params:
        obs_list.append(smpl_params)
        
    if has_limb_weight_params:
        obs_list.append(limb_weight_params)

    obs = torch.cat(obs_list, dim=-1)
    return obs

@torch.jit.script
def compute_imitation_observations_v6(root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel, ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel, time_steps, upright):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor,Tensor,Tensor, int, bool) -> Tensor
    # Adding pose information at the back
    # Future tracks in this obs will not contain future diffs.
    obs = []
    B, J, _ = body_pos.shape

    if not upright:
        root_rot = remove_base_rot(root_rot)

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    heading_rot_expand = heading_rot.unsqueeze(-2).repeat((1, body_pos.shape[1], 1)).repeat_interleave(time_steps, 0)
    

    ##### Body position and rotation differences
    diff_global_body_pos = ref_body_pos.view(B, time_steps, J, 3) - body_pos.view(B, 1, J, 3)
    diff_local_body_pos_flat = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_body_pos.view(-1, 3))

    body_rot[:, None].repeat_interleave(time_steps, 1)
    diff_global_body_rot = torch_utils.quat_mul(ref_body_rot.view(B, time_steps, J, 4), torch_utils.quat_conjugate(body_rot[:, None].repeat_interleave(time_steps, 1)))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), diff_global_body_rot.view(-1, 4)), heading_rot_expand.view(-1, 4))  # Need to be change of basis
    
    ##### linear and angular  Velocity differences
    diff_global_vel = ref_body_vel.view(B, time_steps, J, 3) - body_vel.view(B, 1, J, 3)
    diff_local_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_vel.view(-1, 3))


    diff_global_ang_vel = ref_body_ang_vel.view(B, time_steps, J, 3) - body_ang_vel.view(B, 1, J, 3)
    diff_local_ang_vel = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), diff_global_ang_vel.view(-1, 3))
    

    ##### body pos + Dof_pos This part will have proper futuers.
    local_ref_body_pos = ref_body_pos.view(B, time_steps, J, 3) - root_pos.view(B, 1, 1, 3)  # preserves the body position
    local_ref_body_pos = torch_utils.my_quat_rotate(heading_inv_rot_expand.view(-1, 4), local_ref_body_pos.view(-1, 3))

    local_ref_body_rot = torch_utils.quat_mul(heading_inv_rot_expand.view(-1, 4), ref_body_rot.view(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot)

    # make some changes to how futures are appended.
    obs.append(diff_local_body_pos_flat.view(B, time_steps, -1))  # 1 * timestep * 24 * 3
    obs.append(torch_utils.quat_to_tan_norm(diff_local_body_rot_flat).view(B, time_steps, -1))  #  1 * timestep * 24 * 6
    obs.append(diff_local_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(diff_local_ang_vel.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_pos.view(B, time_steps, -1))  # timestep  * 24 * 3
    obs.append(local_ref_body_rot.view(B, time_steps, -1))  # timestep  * 24 * 6

    obs = torch.cat(obs, dim=-1).view(B, -1)
    return obs
