import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import joblib
import numpy as np
import h5py
from tqdm import tqdm
from collections import defaultdict

if __name__ == "__main__":
    add_action_noise = True
    action_noise_std = 0.05
    dataset_path = "/hdd/zen/dev/meta/EgoQuest/data/amass/pkls/amass_isaac_im_train_take6_upright_slim.pkl"
    # dataset_path = "/hdd/zen/dev/meta/EgoQuest/data/amass/pkls/amass_isaac_run_upright_slim.pkl"
    motion_file_name = dataset_path.split("/")[-1].split(".")[0]
    exp_name = "phc_comp_3"
    dataset_full = joblib.load(dataset_path)
    num_envs = len(dataset_full) if len(dataset_full)  < 512 else 512
    num_runs = 10

    # Creating dataset
    for i in range(num_runs):
        if i == 0:
            add_action_noise = False
        else:
            add_action_noise = True
        cmd = f"python phc/run_hydra.py learning=im_mcp_big  exp_name={exp_name} env=env_im_getup_mcp robot=smpl_humanoid \
        env.zero_out_far=False robot.real_weight_porpotion_boxes=False env.num_prim=3 \
            env.motion_file={dataset_path} env.models=['output/HumanoidIm/phc_3/Humanoid.pth'] \
                env.num_envs={num_envs} headless=True epoch=-1 test=True im_eval=True \
                collect_dataset=True  env.add_action_noise={add_action_noise}   env.action_noise_std={action_noise_std}"
        
        print(cmd)
        os.system(cmd)

    print("Done")
    # Aggregrating the dataset into one file 
    full_dataset = defaultdict(list)
    pkl_files = glob.glob(f"output/HumanoidIm/{exp_name}/phc_act/{motion_file_name}/*.pkl")

    no_concatenate_keys = []

    for file in tqdm(pkl_files):
        file_data = joblib.load(file)
        for k, v in file_data.items():
            full_dataset[k].append(v)

    for key, value in full_dataset.items():
        if key in  ["running_mean"]:
            full_dataset[key] = value[0]
        elif key == "config":
            continue
        else:
            full_dataset[key] = np.concatenate(value, axis=0)

    with h5py.File(f'output/HumanoidIm/{exp_name}/phc_act/phc_act_{motion_file_name}.h5', 'w') as hdf5_file:
        metadata_dump = {}
        for key, value in full_dataset.items():
            # Write each array to the HDF5 file with gzip compression
            print(key)
            if key in ['obs', 'env_action', 'reset', "clean_action", "reset"]:
                h5_dataset = hdf5_file.create_dataset(key, data=value, compression="gzip", compression_opts=9)
            else:
                metadata_dump[key] = value
        joblib.dump(metadata_dump, f'output/HumanoidIm/{exp_name}/phc_act/phc_act_{motion_file_name}_metadata.pkl', compress=True)
                

    import ipdb; ipdb.set_trace()
    
    joblib.dump(full_dataset, f'output/HumanoidIm/{exp_name}/phc_act/phc_act_{motion_file_name}.pkl', compress=True)

