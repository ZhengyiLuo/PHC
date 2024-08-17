import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import joblib

if __name__ == "__main__":
    num_envs = 512
    add_action_noise = True
    action_noise_std = 0.05
    # dataset_path = "/hdd/zen/dev/meta/EgoQuest/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl"
    dataset_path = "/hdd/zen/dev/meta/EgoQuest/data/amass/pkls/amass_isaac_run_upright_slim.pkl"
    exp_name = "phc_comp_3"
    dataset_full = joblib.load(dataset_path)
    num_envs = len(dataset_full) if len(dataset_full)  < 512 else 512
    num_runs = 3

    for i in range(num_runs):
        if i == 0:
            add_action_noise = False
        else:
            add_action_noise = True
        cmd = f"python phc/run_hydra.py learning=im_mcp_big  exp_name={exp_name} env=env_im_getup_mcp robot=smpl_humanoid \
        env.zero_out_far=False robot.real_weight_porpotion_boxes=False env.num_prim=3 \
            env.motion_file={dataset_path} env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] \
                env.num_envs={num_envs} headless=True epoch=-1 test=True im_eval=True \
                env.collect_dataset=True  env.add_action_noise={add_action_noise}   env.action_noise_std={action_noise_std}"
        
        print(cmd)
        os.system(cmd)

    import ipdb; ipdb.set_trace()
    print("Done")