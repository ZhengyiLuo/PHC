# Isaac Lab Evaluation Script Configuration

This document describes the configuration options available for the `eval_in_isaaclab.py` script.

## Overview

The script has been enhanced with configurable options to specify what data to load, including motion files, policy paths, action offset files, and SMPL robot configurations.

## Command Line Arguments

### Basic Configuration
- `--num_envs`: Number of environments to spawn (default: 2)
- `--motion_file`: Path to motion file to load (default: `data/amass/pkls/amass_isaac_run_upright_slim.pkl`)
- `--humanoid_type`: Type of humanoid model to use (choices: `smpl`, `smplx`, default: `smpl`)
- `--num_motions`: Number of motions to load from motion library (default: 10)

### Policy and Action Configuration
- `--policy_path`: Path to policy checkpoint file (optional, uses defaults based on humanoid type)
- `--action_offset_file`: Path to action offset file (optional, uses defaults based on humanoid type)

### SMPL Robot Configuration
- `--smpl_data_dir`: Directory containing SMPL model data (default: `data/smpl`)
- `--use_mesh`: Enable mesh rendering for SMPL robot (flag, default: False)
- `--big_ankle`: Use big ankle configuration for SMPL robot (flag, default: True)
- `--box_body`: Use box body configuration for SMPL robot (flag, default: True)

## Usage Examples

### 1. Basic SMPL Evaluation
```bash
./isaaclab.sh -p scripts/eval_in_isaaclab.py \
    --num_envs 2 \
    --motion_file data/amass/pkls/amass_isaac_run_upright_slim.pkl
```

### 2. SMPLX Evaluation with Custom Policy
```bash
./isaaclab.sh -p scripts/eval_in_isaaclab.py \
    --num_envs 4 \
    --humanoid_type smplx \
    --policy_path output/HumanoidIm/smplx_im_upright_1/Humanoid_00117000.pth
```

### 3. High-Quality SMPL with Mesh Rendering
```bash
./isaaclab.sh -p scripts/eval_in_isaaclab.py \
    --num_envs 8 \
    --use_mesh \
    --big_ankle \
    --box_body \
    --num_motions 50
```

### 4. Minimal Test Configuration
```bash
./isaaclab.sh -p scripts/eval_in_isaaclab.py \
    --num_envs 1 \
    --num_motions 5 \
    --motion_file data/amass/pkls/amass_isaac_standing_upright_slim.pkl
```

### 5. Custom Motion Library
```bash
./isaaclab.sh -p scripts/eval_in_isaaclab.py \
    --num_envs 16 \
    --motion_file data/custom_motions/dance_sequences.pkl \
    --policy_path output/CustomPolicy/dance_policy_001.pth
```

## Default Behavior

When configuration options are not specified, the script falls back to sensible defaults:

- **Policy Path**: Automatically selects based on humanoid type:
  - SMPL: `output/HumanoidIm/phc_3/Humanoid.pth`
  - SMPLX: `output/HumanoidIm/smplx_im_upright_1/Humanoid_00117000.pth`

- **Action Offset File**: Automatically selects based on humanoid type:
  - SMPL: `data/action_offset_smpl.pkl`
  - SMPLX: `data/action_offset_smplx.pkl`

- **SMPL Robot Settings**: Uses optimized defaults for performance and stability

## Configuration File Examples

See `eval_config_example.py` for more detailed configuration examples and usage patterns.

## Notes

- The script automatically handles different humanoid types and their specific requirements
- Mesh rendering can significantly impact performance - use only when needed
- Custom policy paths should be compatible with the specified humanoid type
- Motion files should be in the expected format for the Isaac Lab environment 