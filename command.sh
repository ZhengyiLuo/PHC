#!/bin/bash

# Loop 16 times
for i in {1..8}
do
   echo "Iteration $i"
   # Place your commands here
   python phc/run_hydra.py learning=im_mcp_big  exp_name=phc_comp_3 \
env=env_im_getup_mcp robot=smpl_humanoid env.zero_out_far=False \
robot.real_weight_porpotion_boxes=False env.num_prim=3 \
env.motion_file=./data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl \
env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] env.num_envs=1024 \
headless=True epoch=-1 test=True im_eval=True \
collect_dataset=True  \
env.add_action_noise=True \
env.mlp_bypass=False \
env.action_noise_std=0.05 \
env.start_idx=0
done

python phc/run_hydra.py learning=im_mcp_big  exp_name=phc_comp_3 \
env=env_im_getup_mcp robot=smpl_humanoid env.zero_out_far=False \
robot.real_weight_porpotion_boxes=False env.num_prim=3 \
env.motion_file=./data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl \
env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] env.num_envs=1 \
headless=True epoch=-1 test=True env.mlp_bypass=True 
