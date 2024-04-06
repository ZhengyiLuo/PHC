# SMPL robot

The SMPL robot, an adaption of Robot class from [Transform2Act](https://github.com/Khrylx/Transform2Act), is an automatic humanoid generation class that supports SMPL, SMPL-H, and SMPL-X models. It creates XML file humanoid model for MuJoCo simulator, and can create humanoid of different gender and body shape. It supports both capsule based and mesh based models. We use SMPL Robot to create SMPL models to simulate on-the-fly when training our UHC with different body shapes. To run SMPL Robot, use:

To generate Isaac Gym comptialbe humanoid models, use:

```
python scripts/vis/vis_motion_mj.py
```

This model supports mesh and non-mesh based humanoid used in [Trace & Pace](https://research.nvidia.com/labs/toronto-ai/trace-pace/). There is a number of setting you can play with like "upright_start", which generates a humanoid that that starts in an upright position facing x-direcation (default SMPL faces upward). 


[More to come]
