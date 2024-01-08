## Running the video to control demo. 

To run the video demo, we need to run [Metrabs](https://istvansarandi.com/eccv22_demo/) for pose estimation, which requires tensorflow. 


To install dependency (in addition the already existing ones):
```
conda create -n metrabs python=3.8
conda activate metrabs
pip install asyncio aiohttp  opencv-python scipy joblib charset-normalizer tensorflow-hub pandas ultralytics
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

pip install tensorflow[and-cuda]
```

(I have some trouble installing the newest tensorflow using CUDA 11.6, but 12.1 seems to work fine. )


Then, run 

```
python scripts/demo/video_to_pose_server.py
```

It should run Yolov8 for detection, and run pose estmation using Metrabs. Metrabs may take a little while to load. Make sure you the pose estimation result is visualized before proceeding. You should look for the message
"==================================> Metrabs model loaded <==================================". 

On my 3080, Metrabs can run at 30 FPS. 

Then, run 
```
python phc/run.py --task HumanoidImMCPDemo --cfg_env phc/data/cfg/phc_kp_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_kp_mcp_iccv --test --num_envs 1 --epoch -1  --no_virtual_display
```

If your gpu struggles to run both simulation and pose estimation, you can run the pose estimation and simulation on different machines and communicate via websocket (change the SERVER url in `phc/env/tasks/humanoid_im_mcp_demo.py`). 

**After the server runs, you can type in the command "r:0.5" to change the offset height of the keypoints to make them on the ground.**





