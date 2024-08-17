import joblib
import os
import numpy as np
# Specify the directory you want to search in
directory = './bc_model/obs_clean_actions_reset_action_noise_0.05/0-1024'
 
# Find all .pkl files and record their paths
pkl_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.pkl'):
            pkl_files.append(os.path.join(root, file))
print("pkl files", pkl_files)
all_obs = []
all_actions = []
all_resets =[]
all_length = []
for i in range(4):
    
    # Assuming obs is your list of numpy arrays
    obs, action, reset = joblib.load(pkl_files[i])
    if len(obs)>11313:
        obs = obs[:11313]
        actions = action[:11313]
        reset = reset[:11313]
    # Record the lengths of each array in the list
    length = [arr.shape[0] for arr in obs]
    print("len length", len(length))
    # big_obs = np.empty((0, 934))
    # big_actions=np.empty((0, 69))
    # big_reset = np.empty((0))
    big_obs = []
    big_actions=[]
    big_reset = []
    # Concatenate all arrays into one big array
    for l in range(len(length)):
        assert(obs[l].shape[0]==action[l].shape[0]==len(reset[l]))
        big_obs.append(obs[l])
        big_actions.append(action[l])
        big_reset.append(reset[l])
        # big_reset += reset[l]

    print("big_obs len ", len(big_obs))
    print("big_action len ", len(big_actions))
    print("big_reset len ", len(big_reset))
    big_obs = np.concatenate(big_obs, axis=0)
    big_actions = np.concatenate(big_actions, axis=0)
    big_reset = np.concatenate(big_reset, axis=0)
    print("big obs, ", big_obs.shape)
    print("big action shape ", big_actions.shape)
    print("big reset is ", big_reset.shape)
    all_obs.append(big_obs)
    all_actions.append(big_actions)
    all_resets.append(big_reset)
    all_length.append(length)

all_actions = np.concatenate(all_actions, axis=0)
all_obs = np.concatenate(all_obs, axis=0)
all_resets = np.concatenate(all_resets, axis=0)
all_length = np.array(all_length)
assert(all_actions.shape[0]==all_obs.shape[0]==all_resets.shape[0])
print(all_actions.shape)
print(all_obs.shape)
print(all_resets.shape)
print(all_length.shape)

joblib.dump((all_obs, all_actions, all_resets, all_length),"./bc_model/obs_clean_actions_reset_action_noise_0.05/four0-1024traj.pkl", compress=True)
