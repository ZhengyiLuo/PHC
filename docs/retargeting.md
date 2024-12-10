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

To visualize the fitted motion, you can run the following script. 
```
python scripts/vis/vis_q_mj.py robot=unitree_g1_fitting +motion_name=0-Transitions_mocap_mazen_c3d_dance_stand_poses
```


## Details about the YAML files:


```
extend_config:
  - joint_name: "head_link"
    parent_name: "pelvis"
    pos: [0.0, 0.0, 0.4]
    rot: [1.0, 0.0, 0.0, 0.0]

base_link: "torso_link"
joint_matches:
  - ["pelvis", "Pelvis"]
  - ["left_hip_pitch_link", "L_Hip"]
  - ["left_knee_link", "L_Knee"]
  - ["left_ankle_roll_link", "L_Ankle"]
  - ["right_hip_pitch_link", "R_Hip"]
  - ["right_knee_link", "R_Knee"]
  - ["right_ankle_roll_link", "R_Ankle"]
  - ["left_shoulder_roll_link", "L_Shoulder"]
  - ["left_elbow_pitch_link", "L_Elbow"]
  - ["left_zero_link", "L_Hand"]
  - ["right_shoulder_roll_link", "R_Shoulder"]
  - ["right_elbow_pitch_link", "R_Elbow"]
  - ["right_zero_link", "R_Hand"]
  - ["head_link", "Head"]


smpl_pose_modifier:
  - Pelvis: "[np.pi/2, 0, np.pi/2]"
  - L_Shoulder: "[0, 0, -np.pi/2]"
  - R_Shoulder: "[0, 0, np.pi/2]"
  - L_Elbow: "[0, -np.pi/2, 0]"
  - R_Elbow: "[0, np.pi/2, 0]"
```

The above YAML file is used to setup the retargeting process. 

Extend_config: creating new joints for the humanoid such that there can be more constraints for the motion fitting process (e.g. the head joint is created to be a child of the pelvis joint)

Joint matches: matching the joints between the humanoid and the SMPL model. 

SMPL pose modifier: modifying the SMPL pose to fit reseting pose of the humanoid. For instance, H1's default pose has the elbow bent, so we need to modify the SMPL pose to fit the humanoid. 

