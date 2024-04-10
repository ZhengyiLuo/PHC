import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
from rl_games.algos_torch import torch_ext
import joblib
import numpy as np
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='')
    parser.add_argument('--idx', default=0)
    parser.add_argument('--epoch', default=200000)
    parser.add_argument('--data', default='data/amass/pkls/amass_train_data_take6.pkl')
    
    args = parser.parse_args()
    
    trained_idx = int(args.idx)
    exp_name = args.exp
    epoch = int(args.epoch)
    print(f"PNN Processing for: exp_name: {exp_name}, idx: {trained_idx}, epoch: {epoch}")
    import ipdb; ipdb.set_trace()


    checkpoint = torch_ext.load_checkpoint(f"output/HumanoidIm/{exp_name}/Humanoid_{epoch:08d}.pth")
    amass_train_data_take6 = joblib.load(args.data)

    failed_keys_dict = {}
    termination_history_dict = {}
    all_keys = set()
    for failed_path in sorted(glob.glob(f"output/HumanoidIm/{exp_name}/failed_*"))[:]:
        failed_idx = int(failed_path.split("/")[-1].split("_")[-1].split(".")[0])
        failed_keys_entry = joblib.load(failed_path)
        failed_keys = failed_keys_entry['failed_keys']
        failed_keys_dict[failed_idx] = failed_keys
        termination_history_dict[failed_idx] = failed_keys_entry['termination_history']
        [all_keys.add(k) for k in failed_keys]
        
    dump_keys = []
    for k, v in failed_keys_dict.items():
        if k <= epoch and k >= epoch - 2500 * 5:
            dump_keys.append(v)

    dump_keys = np.concatenate(dump_keys)

    network_name_prefix = "a2c_network.pnn.actors"


    loading_keys = [k for k in checkpoint['model'].keys() if k.startswith(f"{network_name_prefix}.{trained_idx}")] 
    copy_keys = [k for k in checkpoint['model'].keys() if k.startswith(f"{network_name_prefix}.{trained_idx + 1}")] 


    for idx, key_name in enumerate(copy_keys):
        checkpoint['model'][key_name].copy_(checkpoint['model'][loading_keys[idx]])
        
    torch_ext.save_checkpoint(f"output/HumanoidIm/{exp_name}/Humanoid_{epoch + 1:08d}", checkpoint)

    failed_dump = {key: amass_train_data_take6[key] for key in dump_keys if key in amass_train_data_take6}

    os.makedirs(f"data/amass/pkls/auto_pmcp", exist_ok=True)
    print(f"dumping {len(failed_dump)} samples to data/amass/pkls/auto_pmcp/{exp_name}_{epoch}.pkl")
    joblib.dump(failed_dump, f"data/amass/pkls/auto_pmcp/{exp_name}_{epoch}.pkl")
