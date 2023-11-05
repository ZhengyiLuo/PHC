## Quick steps for evaluating in Docker 

1. Download isaacgym replace its Dockerfile in the `docker` with [ours](Dockerfile). 
2. Build a image by running `build.sh`.
3. Run the container by running `run.sh`. 
4. Download the code from this repo by 
```bash
git clone https://github.com/ZhengyiLuo/PerpetualHumanoidControl
```

5. Download SMPL data, rename and put them in the PerpetualHumanoidControl folder
```bash
mv basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl SMPL_NEUTRAL.pkl
mv basicmodel_m_lbs_10_207_0_v1.1.0.pkl SMPL_MALE.pkl
mv basicmodel_f_lbs_10_207_0_v1.1.0.pkl SMPLE_FEMALE.pkl
```

The file sturcture should be: 
```
PerpetualHumanoidControl/
├── data
│   └── smpl
│       ├── SMPL_FEMALE.pkl
│       ├── SMPL_MALE.pkl
│       └── SMPL_NEUTRAL.pkl
```

6. Download the model files:
```bash
sh download.sh
```

7. Now we should be able to run the evaluation script!😄 
```bash
python phc/run.py --task HumanoidImMCPGetup --cfg_env phc/data/cfg/phc_shape_mcp_iccv.yaml --cfg_train phc/data/cfg/train/rlg/im_mcp.yaml --motion_file sample_data/amass_isaac_standing_upright_slim.pkl --network_path output/phc_shape_mcp_iccv --test --num_envs 1 --epoch -1
```

## Dockerfile explained 
We have made several modifications to the official isaacgym Dockerfile.
1. Use `nvcr.io/nvidia/pytorch:23.02-py3` as base image, which ships last version of `1.x` pytorch in nvidia's images. See support matrix [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes). 
2. Add the gymuser to no-password sudoer. 
3. Install `libosmesa6-dev` and `xserver-xephyr` for X11. 
4. Install all python dependency this repo needed. 