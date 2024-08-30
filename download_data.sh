mkdir sample_data
mkdir -p output output/HumanoidIm/ output/HumanoidIm/phc_kp_pnn_iccv output/HumanoidIm/phc_kp_mcp_iccv output/HumanoidIm/phc_shape_mcp_iccv output/HumanoidIm/phc_shape_pnn_iccv output/HumanoidIm/phc_comp_3 output/HumanoidIm/phc_3 output/HumanoidIm/phc_comp_kp_2 output/HumanoidIm/phc_kp_2 output/HumanoidIm/phc_x_pnn
gdown https://drive.google.com/uc?id=1bLp4SNIZROMB7Sxgt0Mh4-4BLOPGV9_U -O  sample_data/ # filtered shapes from AMASS
gdown https://drive.google.com/uc?id=1arpCsue3Knqttj75Nt9Mwo32TKC4TYDx -O  sample_data/ # all shapes from AMASS
gdown https://drive.google.com/uc?id=1fFauJE0W0nJfihUvjViq9OzmFfHo_rq0 -O  sample_data/ # sample standing neutral data.
gdown https://drive.google.com/uc?id=1uzFkT2s_zVdnAohPWHOLFcyRDq372Fmc -O  sample_data/ # amass_occlusion_v3
gdown https://drive.google.com/uc?id=1lROeTwUwZkhzs-NCfzvhFoyJvdy1chPu -O  output/HumanoidIm/phc_kp_pnn_iccv/
gdown https://drive.google.com/uc?id=1eGTO1hm74FIip9m6WzN8a7AWXeTX3SM9 -O  output/HumanoidIm/phc_kp_mcp_iccv/
gdown https://drive.google.com/uc?id=1_B0HgLQElEZhEWkhmg5nweoWKnQB6VYr -O  output/HumanoidIm/phc_shape_pnn_iccv/
gdown https://drive.google.com/uc?id=1g1uXLYPev_2RBUQmP3-uYdtXbN9LKbSL -O  output/HumanoidIm/phc_shape_mcp_iccv/
gdown https://drive.google.com/uc?id=10Y8ZZBi7kQgRjNKRaddDEj8SebPCH7sj -O  output/HumanoidIm/phc_prim_vr/
gdown https://drive.google.com/uc?id=1JbK9Vzo1bEY8Pig6D92yAUv8l-1rKWo3 -O  output/HumanoidIm/phc_comp_3/Humanoid.pth
gdown https://drive.google.com/uc?id=1pS1bRUbKFDp6o6ZJ9XSFaBlXv6_PrhNc -O  output/HumanoidIm/phc_3/Humanoid.pth

gdown https://drive.google.com/uc?id=1V1mG5dTXzkONgPiwd97JeKtnFaj7KwQx -O  output/HumanoidIm/phc_comp_kp_2/Humanoid.pth
gdown https://drive.google.com/uc?id=1QVv1qxsN2LnncPna66qSV3OikYfseCnx -O  output/HumanoidIm/phc_kp_2/Humanoid.pth

gdown https://drive.google.com/uc?id=1wb6mWeTVVWQ9K27NkvJhxO-b4bHpAA5z -O  output/HumanoidIm/phc_x_pnn/Humanoid.pth
gdown https://drive.google.com/uc?id=1vUb7-j_UQRGMyqC_uY0YIdy6May297K5 -O  sample_data/