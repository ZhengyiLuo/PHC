# Dumping state-actions pairs for offline RL (developed by @kangnil)

To dump state-actions pairs for offline RL, you can run the following script. In the script, I hard coded phc_comp_3 as the teacher but you can change it to any other teacher. In that script, I dump state-action pairs with action noise. 

```
python scripts/phc_act/create_phc_act_dataset.py --dataset_path=data/amass/pkls/amass_isaac_run_upright_slim.pkl --exp_name=phc_comp_3 --num_runs=10 --action_noise_std=0.05
```


To train the a BC policy using the dumped state-actions pairs, you can run the following script. 
```
python scripts/phc_act/train_phc_actor.py --dataset_path=output/HumanoidIm/phc_comp_3/phc_act/phc_act_amass_isaac_run_upright_slim.pkl --metadata_path=output/HumanoidIm/phc_comp_3/phc_act/phc_act_amass_isaac_run_upright_slim_meta_data.pkl --output_path=output/HumanoidIm/phc_comp_3/phc_act/models/
```


To evaluate the BC policy, you can run the following script. 

```
python phc/run_hydra.py learning=im_mcp_big  exp_name=phc_comp_3 env=env_im_getup_mcp robot=smpl_humanoid env.zero_out_far=False robot.real_weight_porpotion_boxes=False env.num_prim=3 env.motion_file=data/amass/pkls/amass_isaac_run_upright_slim.pkl env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] env.num_envs=1 headless=False epoch=-1 test=True env.mlp_bypass=True env.mlp_model_path=[insert path to trained BC model]
```

For running sequences (~250 sequences), 100 epoch reaches 97% success rate. 