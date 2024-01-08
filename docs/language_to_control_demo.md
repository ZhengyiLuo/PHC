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
python phc/run.py --task HumanoidImMCPDemo --cfg_env phc/data/cfg/phc_kp_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_kp_mcp_iccv --test --num_envs 1 --epoch -1  --no_virtual_display
```