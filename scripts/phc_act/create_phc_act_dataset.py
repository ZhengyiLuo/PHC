import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())


if __name__ == "__main__":
    parent_folder = "./bc_model/obs_clean_actions_reset_action_noise_0.05/"
    pkl_files = [parent_folder+f for f in os.listdir(parent_folder) if f.endswith('.pkl')]
    print(pkl_files) 
    file_name = "first128"
    foldername = os.path.join(parent_folder, file_name)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
 
    train(pkl_files, foldername)