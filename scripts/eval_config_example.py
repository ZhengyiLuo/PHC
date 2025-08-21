#!/usr/bin/env python3
"""
Example configuration for eval_in_isaaclab.py

This file demonstrates how to use the various configuration options
available for the Isaac Lab evaluation script.
"""

# Example 1: Basic SMPL configuration
BASIC_SMPL_CONFIG = {
    "motion_file": "data/amass/pkls/amass_isaac_run_upright_slim.pkl",
    "humanoid_type": "smpl",
    "num_motions": 10,
    "num_envs": 2
}

# Example 2: SMPLX configuration with custom paths
SMPLX_CONFIG = {
    "motion_file": "data/amass/pkls/amass_isaac_dance_slim.pkl",
    "policy_path": "output/HumanoidIm/smplx_im_upright_1/Humanoid_00117000.pth",
    "action_offset_file": "data/action_offset_smplx_custom.pkl",
    "humanoid_type": "smplx",
    "num_motions": 20,
    "num_envs": 4
}

# Example 3: High-quality SMPL configuration with mesh rendering
HIGH_QUALITY_SMPL_CONFIG = {
    "motion_file": "data/amass/pkls/amass_isaac_walk_upright_slim.pkl",
    "policy_path": "output/HumanoidIm/phc_3/Humanoid.pth",
    "action_offset_file": "data/action_offset_smpl_optimized.pkl",
    "humanoid_type": "smpl",
    "num_motions": 50,
    "num_envs": 8,
    "use_mesh": True,
    "big_ankle": True,
    "box_body": True,
    "smpl_data_dir": "data/smpl_high_quality"
}

# Example 4: Minimal configuration for testing
MINIMAL_CONFIG = {
    "motion_file": "data/amass/pkls/amass_isaac_standing_upright_slim.pkl",
    "humanoid_type": "smpl",
    "num_motions": 5,
    "num_envs": 1
}

# Example 5: Custom motion library configuration
CUSTOM_MOTION_CONFIG = {
    "motion_file": "data/custom_motions/dance_sequences.pkl",
    "policy_path": "output/CustomPolicy/dance_policy_001.pth",
    "action_offset_file": "data/custom_action_offsets.pkl",
    "humanoid_type": "smpl",
    "num_motions": 100,
    "num_envs": 16,
    "use_mesh": False,
    "big_ankle": False,
    "box_body": False,
    "smpl_data_dir": "data/smpl_custom"
}

def print_config_usage():
    """Print example command line usage for different configurations."""
    print("Example Usage Commands:")
    print("=" * 60)
    
    print("\n1. Basic SMPL evaluation:")
    print("./isaaclab.sh -p scripts/eval_in_isaaclab.py --num_envs 2 --motion_file data/amass/pkls/amass_isaac_run_upright_slim.pkl")
    
    print("\n2. SMPLX evaluation with custom policy:")
    print("./isaaclab.sh -p scripts/eval_in_isaaclab.py --num_envs 4 --humanoid_type smplx --policy_path output/HumanoidIm/smplx_im_upright_1/Humanoid_00117000.pth")
    
    print("\n3. High-quality SMPL with mesh rendering:")
    print("./isaaclab.sh -p scripts/eval_in_isaaclab.py --num_envs 8 --use_mesh --big_ankle --box_body --num_motions 50")
    
    print("\n4. Minimal test configuration:")
    print("./isaaclab.sh -p scripts/eval_in_isaaclab.py --num_envs 1 --num_motions 5 --motion_file data/amass/pkls/amass_isaac_standing_upright_slim.pkl")
    
    print("\n5. Custom motion library:")
    print("./isaaclab.sh -p scripts/eval_in_isaaclab.py --num_envs 16 --motion_file data/custom_motions/dance_sequences.pkl --policy_path output/CustomPolicy/dance_policy_001.pth")

if __name__ == "__main__":
    print_config_usage() 