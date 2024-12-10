import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import joblib
import random
import os
import wandb
import argparse
from datetime import datetime
from phc.learning.mlp import MLP
import h5py
from tqdm import tqdm
import glob
import os
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import joblib
import numpy as np
import random
import argparse

# wandb.login()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class HumanoidDataset(Dataset):
    def __init__(self, dataset_path, metadata_path, use_pretrained_running_mean = True, sample_pkl=False, directory=""):
        if sample_pkl and os.path.exists(directory):
            all_pkl_files = glob.glob(f"{directory}/*.pkl")

            noise_0_005 = []
            noise_0075_01 = []

            for pkl_file in all_pkl_files:
                split = pkl_file.split("/")[-1].split("_")
                if split[1] == "False" or split[2] == "0.05":
                    noise_0_005.append(pkl_file)
                elif split[2]=="0.075":
                    noise_0075_01.append(pkl_file)

            pkl_files = random.sample(noise_0_005, 5) + random.sample(noise_0075_01, 1)
            print("curr dataset: ", pkl_files)

            data = defaultdict(list)

            for file in tqdm(pkl_files):
                file_data = joblib.load(file)
                for k, v in file_data.items():
                    data[k].append(v)

            for key, value in data.items():
                if key in ["running_mean"]:
                    data[key] = value[0]
                elif key == "config":
                    continue
                else:
                    data[key] = np.concatenate(value, axis=0)
        else:
            data = joblib.load(dataset_path)


        meta_data = joblib.load(metadata_path)
        self.obs = torch.tensor(data['obs'])
        self.actions = torch.tensor(data['clean_action'])

        if use_pretrained_running_mean:
            self.running_mean = meta_data["running_mean"]["running_mean"].cpu().float()
            self.running_var = meta_data["running_mean"]["running_var"].cpu().float()
            self.obs.sub_(self.running_mean).div_(torch.sqrt(self.running_var + 1e-05))
            self.obs.clamp_(min=-5.0, max=5.0)

        self.obs_size = self.obs.shape[-1]
        self.action_size = self.actions.shape[-1]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]
    
def save_checkpoint(model, optimizer, epoch, loss, foldername, filename="checkpoint.pth"):
    checkpoint_path = os.path.join(foldername, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

def load_checkpoint(model, optimizer, foldername, ckpt="checkpoint.pth"):
    checkpoint_path = os.path.join(foldername, ckpt)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            if isinstance(state, torch.Tensor):
                state.data = state.data.to(device)
            elif isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}")
        return model, optimizer, start_epoch, loss
    else:
        print("No checkpoint found. Starting training from scratch.")
        return model, optimizer, 0, None 

def sample_pkl(directory):
    # Use glob to list all .pkl files in the directory
    all_pkl_files = glob.glob(f"{directory}/*.pkl")

    noise_0_005 = []
    noise_0075_01 = []

    for pkl_file in all_pkl_files:
        split = pkl_file.split("/")[-1].split("_")
        if split[1] == "False" or split[2] == "0.05":
            noise_0_005.append(pkl_file)
        else:
            noise_0075_01.append(pkl_file)

    pkl_files = random.sample(noise_0_005, 5) + random.sample(noise_0075_01, 1)
    print("curr dataset: ", pkl_files)

    full_dataset = defaultdict(list)

    for file in tqdm(pkl_files):
        file_data = joblib.load(file)
        for k, v in file_data.items():
            full_dataset[k].append(v)

    for key, value in full_dataset.items():
        if key in ["running_mean"]:
            full_dataset[key] = value[0]
        elif key == "config":
            continue
        else:
            full_dataset[key] = np.concatenate(value, axis=0)

    time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    save_file = f'{directory}/combined_pkl'
    os.makedirs(save_file, exist_ok=True)
    with open(f'{save_file}/{time}.pkl', 'wb') as f:
        joblib.dump(full_dataset, f)

    with open(f'{save_file}/{time}.txt', 'w') as f:
        for file in pkl_files:
            f.write(f"{file}\n")
    del full_dataset
    return f'{save_file}/{time}.pkl'

def train_model(model, device, criterion, optimizer, batch_size,
                dataset_path, metadata_path, sample_pkl,
                start_epoch, num_epochs, foldername, save_frequency = 100):
    model.to(device)
    pbar = tqdm(range(start_epoch, num_epochs))
    if not sample_pkl:
        dataset = HumanoidDataset(dataset_path, metadata_path)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    for epoch in pbar:
        if epoch%save_frequency == 0 and sample_pkl:
            dataset = HumanoidDataset(dataset_path, metadata_path, sample_pkl = sample_pkl,
                                      directory="output/HumanoidIm/phc_comp_3/phc_act/amass_isaac_im_train_take6_upright_slim/")
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)
        for batch_obs, batch_actions in data_loader:
            # Forward pass
            batch_obs, batch_actions = batch_obs.to(device), batch_actions.to(device)
             
            outputs = model(batch_obs)
            loss = criterion(outputs, batch_actions)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not wandb.run is None:
            wandb.log({"loss": loss.item()})

        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
        if (epoch + 1) % save_frequency == 0:
            #torch.save(model.state_dict(), f'{foldername}/{epoch+1:05d}.pth')
            save_checkpoint(model, optimizer, epoch, loss, foldername, filename=f"{epoch+1:05d}.pth")
            if sample_pkl:
                del data_loader
                del dataset
    print("Training complete.")
    return model



def train(dataset_path, metadata_path, sample_pkl, output_path, ckpt_path="checkpoint.pth"):
    units = [2048, 1024, 512]  # Example hidden layer size
    batch_size = 16384
    num_epochs = 100000
    save_frequency = 100
    learning_rate = 2e-5

    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="PHC_Act",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": learning_rate,
    #         "hidden_size": units,
    #         "batch_size": batch_size,
    #         "num_epochs": num_epochs,
    #         "dataset": dataset_path,
    #     },
    # )
    model = MLP(934, 69, units, "silu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, output_path, ckpt=ckpt_path)

    # Train the model
    train_model(model, device, criterion, optimizer, batch_size, dataset_path, metadata_path, sample_pkl, start_epoch, num_epochs, output_path, save_frequency = save_frequency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--metadata_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--sample_pkl", type=bool, default=False)
    parser.add_argument("--ckpt_path", type=str, default="01600.pth")
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    metadata_path = args.metadata_path
    output_path = args.output_path
    ckpt_path = args.ckpt_path
    os.makedirs(output_path, exist_ok=True)
    sample_pkl = args.sample_pkl
    train(dataset_path, metadata_path, sample_pkl, output_path, ckpt_path)

