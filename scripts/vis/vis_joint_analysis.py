import os
import sys
import time
import argparse
import os.path as osp
sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree
import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot
import mujoco
import mujoco.viewer
import joblib
import hydra
from omegaconf import DictConfig

def get_movement_category(movement):
    if movement > 1.0:
        return "HIGH", "\033[91m"  # Red
    elif movement > 0.3:
        return "MEDIUM", "\033[93m"  # Yellow
    elif movement > 0.1:
        return "LOW", "\033[94m"  # Blue
    else:
        return "NONE", "\033[90m"  # Gray

def key_call_back(keycode):
    global curr_start, motion_id, time_step, dt, paused, isolated_joint, joint_group, joint_groups, joint_group_names
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "N":
        # Next joint
        isolated_joint = (isolated_joint + 1) % total_dofs
        print(f"Showing joint {isolated_joint}: {dof_names[isolated_joint] if isolated_joint < len(dof_names) else 'Unknown'}")
        time_step = 0
    elif chr(keycode) == "P":
        # Previous joint
        isolated_joint = (isolated_joint - 1) % total_dofs
        print(f"Showing joint {isolated_joint}: {dof_names[isolated_joint] if isolated_joint < len(dof_names) else 'Unknown'}")
        time_step = 0
    elif chr(keycode) == "A":
        # Show all joints
        isolated_joint = -1
        print("Showing all joints")
        time_step = 0
    elif chr(keycode) == "G":
        # Cycle through joint groups
        joint_group = (joint_group + 1) % len(joint_groups)
        isolated_joint = -2  # Special flag for group mode
        print(f"Showing joint group: {joint_group_names[joint_group]}")
        time_step = 0

@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    global curr_start, motion_id, time_step, dt, paused, isolated_joint, total_dofs, dof_names, joint_group, joint_groups, joint_group_names
    device = torch.device("cpu")
    
    # Initialize globals
    curr_start, motion_id, time_step, dt, paused = 0, 0, 0, 1/30, False
    isolated_joint = -1  # -1 means show all joints
    joint_group = 0
    
    # Load motion data
    motion_file = f"data/{cfg.robot.humanoid_type}/v1/singles/{cfg.motion_name}.pkl"
    print(f"Loading motion from: {motion_file}")
    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())
    curr_motion_key = motion_data_keys[0]
    curr_motion = motion_data[curr_motion_key]
    
    # Get DOF names from the data
    humanoid_xml = cfg.robot.asset.assetFileName
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    
    # Get total DOFs and try to get DOF names
    total_dofs = curr_motion['dof'].shape[1]
    try:
        dof_names = [name for name in curr_motion.get('bfs_dof_names', [])]
        print(f"\nFound {len(dof_names)} named DOFs out of {total_dofs} total DOFs")
    except:
        dof_names = [f"DOF_{i}" for i in range(total_dofs)]
    
    # Analyze movement for each DOF
    dof_data = curr_motion['dof']
    movement = np.ptp(dof_data, axis=0)
    std_dev = np.std(dof_data, axis=0)
    
    # Group joints by movement range
    high_movement = np.where(movement > 1.0)[0]
    medium_movement = np.where((movement > 0.3) & (movement <= 1.0))[0]
    low_movement = np.where((movement > 0.1) & (movement <= 0.3))[0]
    
    joint_groups = [high_movement, medium_movement, low_movement]
    joint_group_names = ["High Movement (>1.0 rad)", "Medium Movement (0.3-1.0 rad)", "Low Movement (0.1-0.3 rad)"]
    
    # Print movement analysis with color coding
    print("\nDOF Movement Analysis:")
    print("Index | DOF Name | Movement Category | Range of Motion | Std Dev")
    print("-" * 75)
    for i, (mov, std) in enumerate(zip(movement, std_dev)):
        category, color = get_movement_category(mov)
        dof_name = dof_names[i] if i < len(dof_names) else f"DOF_{i}"
        print(f"{color}{i:3d} | {dof_name:30s} | {category:15s} | {mov:10.6f} | {std:10.6f}\033[0m")
    
    # Print joint groups
    for group_idx, (group, name) in enumerate(zip(joint_groups, joint_group_names)):
        print(f"\n{name}:")
        for idx in group:
            dof_name = dof_names[idx] if idx < len(dof_names) else f"DOF_{idx}"
            print(f"  Index {idx}: {dof_name} (movement: {movement[idx]:.6f})")
    
    print("\nControls:")
    print("SPACE - Pause/Resume")
    print("R     - Reset to beginning")
    print("N     - Next joint")
    print("P     - Previous joint")
    print("A     - Show all joints")
    print("G     - Cycle through joint groups")
    
    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Get current frame
            curr_time = int(time_step/dt) % curr_motion['dof'].shape[0]
            
            # Set root position and rotation
            mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
            
            # Set joint positions based on mode
            joint_positions = np.zeros_like(curr_motion['dof'][curr_time])
            if isolated_joint == -1:
                # Show all joints
                joint_positions = curr_motion['dof'][curr_time]
            elif isolated_joint == -2:
                # Show current joint group
                for idx in joint_groups[joint_group]:
                    joint_positions[idx] = curr_motion['dof'][curr_time][idx]
            else:
                # Show single joint
                joint_positions[isolated_joint] = curr_motion['dof'][curr_time][isolated_joint]
            
            mj_data.qpos[7:] = joint_positions
            mujoco.mj_forward(mj_model, mj_data)
            
            if not paused:
                time_step += dt
            
            # Visualize SMPL joints if available
            if 'smpl_joints' in curr_motion:
                joint_gt = curr_motion['smpl_joints']
                for i in range(joint_gt.shape[1]):
                    viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
            
            viewer.sync()
            
            # Maintain timing
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main() 