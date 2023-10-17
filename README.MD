
# Perpetual Humanoid Control for Real-time Simulated Avatars

Official implementation of ICCV 2023 paper: "Perpetual Humanoid Control for Real-time Simulated Avatars". In this paper, we present a physics-based humanoid controller that achieves high-fidelity motion imitation and fail-statue recovery in the presence of noisy input (e.g. pose estimates from video or generated from language) and unexpected falls. No external forces is used. 

[[paper]](https://arxiv.org/abs/2305.06456) [[website]](https://zhengyiluo.github.io/PHC/) [[Video]](https://www.youtube.com/watch?v=zS6Y00EW37A)

<div float="center">
  <img src="assets/phc.gif" />
</div>

## News 🚩

[October 3, 2023] Training and Evaluation code released.


## TODOs

- [ ] Add support for smplx/h (fingers!!!).

- [ ] Release PHC+ model (100% success rate on AMASS) used in [PULSE](https://zhengyiluo.github.io/PULSE/). 

- [ ] Release vr controller tracking code.  

- [ ] Release image and language-based demo code.  

- [x] Release training code. 

- [x] Release evaluation code. 

## Introduction
We present a physics-based humanoid controller that achieves high-fidelity motion imitation and fault-tolerant behavior in the presence of noisy input (e.g. pose estimates from video or generated from language) and unexpected falls. Our controller scales up to learning ten thousand motion clips without using any external stabilizing forces and learns to naturally recover from fail-state. Given reference motion, our controller can perpetually control simulated avatars without requiring resets. At its core, we propose the progressive multiplicative control policy (PMCP), which dynamically allocates new network capacity to learn harder and harder motion sequences. PMCP allows efficient scaling for learning from large-scale motion databases and adding new tasks, such as fail-state recovery, without catastrophic forgetting. We demonstrate the effectiveness of our controller by using it to imitate noisy poses from video-based pose estimators and language-based motion generators in a live and real-time multi-person avatar use case.

## Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:


```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 


3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. Please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl

```


Make sure you have the SMPL paramters properly setup by running the following scripts:
```
python uhc/smpllib/smpl_local_robot.py
python scripts/joint_monkey_smpl.py
```



## Data processing for evaluating/training PHC

### Evaluation
Use the following script to download trained models and sample data.

```
bash download_data.sh
```

### Training

We train on a subset of the [AMASS](https://amass.is.tue.mpg.de/) dataset.

For prcessing the AMASS, first, download the AMASS dataset from [AMASS](https://amass.is.tue.mpg.de/). Then, run the following script on the unzipped data:


```
python scripts/data_process/process_amass_raw.py --dir [path_to_amass] --out_dir [out_dir]
```

which dumps the data into the `amass_db_smplh.pt` file. Then, run 

```
python scripts/data_process/process_amass_db.py
```

We further process these data into Motionlib format by running the following script:

```
python scripts/data_process/convert_amass_isaac.py
```

Which should give you 11313 sequences for trainig. 

## Evaluation 


### Imitation 

Evaluate full model:

```
## Shape + rotation + keypoint model
python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_shape_mcp_iccv --test --num_envs 1 --epoch -1

## keypoint model
python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_kp_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_kp_mcp_iccv --test --num_envs 1 --epoch -1
```

Evaluate single primitive model:

```
## Shape + rotation + keypoint model
python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_shape_pnn_iccv --test --num_envs 1 --epoch -1

## keypoint model
python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_kp_pnn_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_kp_pnn_iccv --test --num_envs 1 --epoch -1
```

### In-the-wild Avatar Control

[More instructions coming soon.]

see `humanoid_im_mcp_demo.py`

```
python phc/run.py --task HumanoidImMCPDemo --cfg_env phc/data/cfg/phc_kp_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_kp_mcp_iccv --test --num_envs 1 --epoch -1 

```


## Training

[This section is still under contruction. Proceed with caution.]

Training PHC is not super automated yet, so it requires some (a lot of) manual steps, and invovles changing the config file a couple of times during training based on the training phase. The `phc_shape_pnn_train_iccv.yaml` config file provides a starting point for training primitives. 

First, we will train one primitive, and keep an eye on its performance (--has_eval) flag. In the config, the "training_prim" is the primitive that is being trained. This need to be updated accordingly. 

```
python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml --motion_file data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl --network_path output/phc_shape_pnn_train_iccv --headless --has_eval
```

After the performance plateaus, we will dump the most recent sequences that the primitives has failed on, and use them to train the next primitive. Here idx is the primitive that should be trained. 

```
python scripts/pmcp/forward_pmcp.py --exp phc_shape_mcp_iccv --epoch {} --idx {idx}
```
The above script will dump two files: one is the next hard sequences to learn, and anothere one is the checkpoint to resume with the copied primitive. In this step, we need to modify the config file's two fields: "training_prim" and "fitting", they should look like: 

training_prim: 1 # advancing the primitive to train. 
fitting: True # Fitting flag freezes the running mean and std. This is very important for resuable primitives. 

```
python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml --motion_file data/amass/pkls/auto_pmcp/{exp_name}_{epoch}.pkl --network_path output/phc_shape_pnn_train_iccv --headless --has_eval --epoch {epoch} 
```

Repeat this process until no hard sequences are left. Then, to train the fail-state recovery primitive on simple locomotion data, change the following flags: 
training_prim: ++1 # advancing the primitive to train. 
zero_out_far: True
zero_out_far_train: True
getup_udpate_epoch: {epoch}


```
python phc/run.py --task HumanoidImGetup --cfg_env phc/data/cfg/phc_shape_pnn_train_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_pnn.yaml --motion_file data/amass/pkls/amass_isaac_run_take6_upright_slim.pkl --network_path output/phc_shape_pnn_train_iccv --headless --has_eval --epoch {epoch} 
```


After all primitives are trained, train the composer: 

```
python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_shape_mcp_iccv --has_eval --headless 
```

When training the composer, you can repeat the process above (progressive mining hard sequences) to improve performance. 


You can also just train one model for imitation (no PNN):
```
python phc/run.py --task HumanoidIm --cfg_env phc/data/cfg/phc_prim_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im.yaml --motion_file data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl --network_path output/phc_prim_iccv --headless --has_eval
```


## Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| f | focus on humanoid |
| Right click + WASD | change view port |
| Shift + Right click + WASD | change view port fast |
| r | reset episode |
| j | apply large force to the humanoid |
| l | record screenshot, press again to stop recording|
| ; | cancel screen shot|
| m | cancel termination based on imitation |

... more short cut can be found in `pacer/env/tasks/base_task.py`

Notes on rendering: I am using pyvirtualdisplay to record the video such that you can see all humanoids at the same time (default function will only capture the first environment). You can disable it using the flag `--no_virtual_display`. 


## Citation
If you find this work useful for your research, please cite our paper:
```
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}            
```

Also consider citing these prior works that are used in this project:

```
@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}     

@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

```

## References
This repository is built on top of the following amazing repositories:
* Main code framework is from: [AMP](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
* Part of the SMPL_robot code is from: [UHC](https://github.com/ZhengyiLuo/UniversalHumanoidControl)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)

Please follow the lisence of the above repositories for usage. 
