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

wandb.login()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class HumanoidDataset(Dataset):
    def __init__(self, dataset_path, use_pretrained_running_mean = True):
        meta_data = joblib.load(dataset_path + "_meta_data.pkl")
        self.hdf5_files = h5py.File(dataset_path + ".h5", 'r') 
        self.obs = torch.tensor(self.hdf5_files["obs"][:])
        self.actions = torch.tensor(self.hdf5_files["clean_action"][:])

        if use_pretrained_running_mean:
            self.running_mean = meta_data["running_mean"]["running_mean"].cpu().float()
            self.running_var = meta_data["running_mean"]["running_var"].cpu().float()
            self.obs = (self.obs - self.running_mean) / torch.sqrt(self.running_var + 1e-05)
            self.obs = torch.clamp(self.obs, min=-5.0, max=5.0)

        self.obs_size = self.obs.shape[-1]
        self.action_size = self.actions.shape[-1]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]
    
    def __del__(self):
        self.hdf5_file.close()


def train_model(model, device, criterion, optimizer, data_loader, num_epochs, foldername, save_frequency = 100):
    model.to(device)
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
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
            torch.save(model.state_dict(), f'{foldername}/{epoch+1:05d}.pth')

    print("Training complete.")
    return model

def train(dataset_path, output_path):
    units = [2048, 1024, 512]  # Example hidden layer size
    batch_size = 16384
    num_epochs = 1000
    save_frequency = 100
    learning_rate = 2e-5

    dataset = HumanoidDataset(dataset_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    run = wandb.init(
        # Set the project where this run will be logged
        project="PHC_Act",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "hidden_size": units,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "dataset": dataset_path,
        },
    )

    # Instantiate the model, define the loss function and the optimizer
    model = MLP(dataset.obs_size, dataset.action_size, units, "silu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, device, criterion, optimizer, data_loader, num_epochs, output_path, save_frequency = save_frequency)


if __name__ == "__main__":
    dataset_path = "output/HumanoidIm/phc_comp_3/phc_act/phc_act_amass_isaac_run_upright_slim"
    output_path = "output/HumanoidIm/phc_comp_3/phc_act/models/"
    os.makedirs(output_path, exist_ok=True)
    train(dataset_path, output_path)

