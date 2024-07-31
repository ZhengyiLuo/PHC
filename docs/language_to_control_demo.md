## Running the language to control demo. 

To run the video demo, we need to run [MDM](https://guytevet.github.io/mdm-page/) for generating motion from language. 


To start, download and setup [my forked MDM repository](https://github.com/ZhengyiLuo/motion-diffusion-model)


After setup, go to MDM directory and run 

```
python language_to_pose_server.py --model_path [MDM model path] 
```

Then, you can start typing the language commands in the server script after the "Type MDM Prompt:" message. Press "return" to send more commands. 

Run the following command to start the simulation. 

```
python phc/run_hydra.py learning=im_mcp exp_name=phc_kp_mcp_iccv env=env_im_getup_mcp env.task=HumanoidImMCPDemo robot=smpl_humanoid robot.freeze_hand=True robot.box_body=False env.z_activation=relu env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] env.num_envs=1 env.obs_v=7 headless=False epoch=-1 test=True no_virtual_display=True
```