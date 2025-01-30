import os
import sys
import time
import pandas as pd
import numpy as np
sys.path.append(os.getcwd())

import mujoco
import joblib
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import mujoco.viewer

def init_mujoco_viewer(model, data):
    """Initialize Mujoco viewer using GLContext"""
    width = 1200
    height = 900
    ctx = mujoco.GLContext(width, height)  # Create context with window dimensions
    ctx.make_current()
    vopt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=model.ngeom * 2)
    cam = mujoco.MjvCamera()
    cam.distance = 4.
    cam.azimuth = 90
    cam.elevation = -20
    pert = mujoco.MjvPerturb()
    viewport = mujoco.MjrRect(0, 0, width, height)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    return ctx, vopt, scn, cam, pert, viewport, context

def render(model, data, ctx, vopt, scn, cam, pert, viewport, context):
    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(
        model, data, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scn
    )
    mujoco.mjr_render(viewport, scn, context)
    ctx.swap()

def analyze_joint_hierarchy(model):
    print("\nAnalyzing joint hierarchy:")
    print("-" * 80)
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = mujoco.mjtJoint(model.jnt_type[i]).name
        qpos_adr = model.jnt_qposadr[i]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[i])
        
        print(f"Joint {i:3d}: {jnt_name:30s} | Type: {jnt_type:6s} | QPos Addr: {qpos_adr:3d} | Parent: {parent_name}")
        
        if "right_" in jnt_name and "finger" not in jnt_name and "hand" not in jnt_name:
            dof_range = model.jnt_range[i]
            axis = model.jnt_axis[i]
            print(f"  → DOF range: [{dof_range[0]:5.2f} {dof_range[1]:5.2f}]")
            print(f"  → Axis: [{axis[0]:.1f} {axis[1]:.1f} {axis[2]:.1f}]")
            print(f"  → DOF address: {qpos_adr}")

@hydra.main(version_base=None, config_path="../../phc/data/cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    motion_name = cfg.get("motion_name", "0-wave_141_16_poses")
    motion_path = Path(f"data/{cfg.robot.humanoid_type}/v1/singles/{motion_name}.pkl")
    print(f"\nLoading motion from: {motion_path}")
    
    model = mujoco.MjModel.from_xml_path(cfg.robot.asset.assetFileName)
    data = mujoco.MjData(model)
    
    print("\nAnalyzing joint hierarchy:")
    print("-" * 80)
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = mujoco.mjtJoint(model.jnt_type[i]).name
        qpos_adr = model.jnt_qposadr[i]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[i])
        
        print(f"Joint {i:3d}: {jnt_name:30s} | Type: {jnt_type:6s} | QPos Addr: {qpos_adr:3d} | Parent: {parent_name}")
        
        if "right_" in jnt_name and "finger" not in jnt_name and "hand" not in jnt_name:
            dof_range = model.jnt_range[i]
            axis = model.jnt_axis[i]
            print(f"  → DOF range: [{dof_range[0]:5.2f} {dof_range[1]:5.2f}]")
            print(f"  → Axis: [{axis[0]:.1f} {axis[1]:.1f} {axis[2]:.1f}]")
            print(f"  → DOF address: {qpos_adr}")
    
    motion_data = joblib.load(motion_path)
    curr_motion = motion_data[list(motion_data.keys())[0]]
    dof_data = curr_motion['dof']
    n_frames = len(dof_data)
    n_dofs = dof_data.shape[1]
    
    right_arm_joints = {
        'right_arm_link': 47,
        'right_shoulder_pitch_link': 48,
        'right_shoulder_roll_link': 49,
        'right_shoulder_yaw_link': 50,
        'right_elbow_link': 51,
        'right_wrist_roll_link': 52,
        'right_wrist_pitch_link': 53,
        'right_wrist_yaw_link': 54
    }
    
    joint_data = {}
    print("\nDOF Movement Analysis:")
    print("Index | Movement Range | Std Dev")
    print("-" * 80)
    
    for i in range(n_dofs):
        values = dof_data[:, i]
        movement_range = max(values) - min(values)
        std_dev = np.std(values)
        print(f"{i:3d} | {movement_range:12.6f} | {std_dev:8.6f}")
        
        for joint_name, joint_idx in right_arm_joints.items():
            if i == joint_idx:
                print(f"Using {joint_name} at index {i} with movement range: {movement_range:.4f}")
                joint_data[joint_name] = values
    
    df = pd.DataFrame(joint_data)
    csv_filename = f"right_arm_movements_{motion_name}.csv"
    df.to_csv(csv_filename, index=False)
    
    print(f"\nSaved joint movements to {csv_filename}")
    print(f"Shape: {df.shape} (frames × joints)")
    print("\nColumns (joints):")
    for col in df.columns:
        print(f"  {col}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nMovement statistics for included joints:")
    for joint in df.columns:
        movement_range = df[joint].max() - df[joint].min()
        std_dev = df[joint].std()
        print(f"{joint:30s} | Range: {movement_range:8.4f} | Std: {std_dev:8.4f}")
    
    moving_joints = [joint for joint in df.columns if df[joint].std() > 0.01]
    filtered_df = df[moving_joints]
    filtered_csv = f"right_arm_moving_joints_{motion_name}.csv"
    filtered_df.to_csv(filtered_csv, index=False)
    
    print(f"\nSaved filtered joint movements to {filtered_csv}")
    print(f"Shape: {filtered_df.shape} (frames × moving joints)")
    print("\nMoving joints:")
    for joint in moving_joints:
        print(f"  {joint}")
    
    print("\nStarting visualization... Press Ctrl+C to exit")
    print("Original motion on the left, isolated right arm on the right")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20
        
        try:
            while viewer.is_running():
                for frame in range(n_frames):
                    # Zero out all joints first
                    data.qpos[7:] = 0.0
                    
                    # Only set the right arm joint values
                    for joint_name, joint_idx in right_arm_joints.items():
                        data.qpos[joint_idx + 7] = dof_data[frame, joint_idx]
                    
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")

if __name__ == "__main__":
    main() 