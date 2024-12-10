# Retargeting to Your Own Humanoids
To retarget SMPL data to your own humanoids, you need to follow the following steps: 

First, download and setup the SMPL body models as well as AMASS data (code is provided in the [README.md](../README.md)).

Then, run the following scripts to fit the SMPL motion and shape to your humanoids. We will use unitree_g1 as an example. 

First, we will find the SMPL shape parameters that best fit the humanoid. 
```
python scripts/data_process/fit_smpl_shape.py robot=unitree_g1_fitting
```

Then, we will find the SMPL motion parameters that best fit the humanoid. (pass in `+fit_all=True` to fit all the motion clips in AMASS, otherwise only fit one motion)
```
python scripts/data_process/fit_smpl_motion.py robot=unitree_g1_fitting +amass_root=/path/to/your/amass/data 
```

```
python scripts/vis/vis_q_mj.py robot=unitree_g1_fitting
```

