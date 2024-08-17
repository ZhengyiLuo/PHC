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

wandb.login()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class HumanoidDataset(Dataset):
    def __init__(self, obs, actions, length):
        self.obs = obs
        self.actions = actions
        self.length = length
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        # if self.length > self.obs[idx].shape[0]:
        #     print("idx is ", idx, self.obs[idx].shape)
        #     import pdb; pdb.set_trace()
        rand_index = random.randint(0, self.obs[idx].shape[0] - self.length)
        obs = self.obs[idx][rand_index:rand_index+self.length]
        action = self.actions[idx][rand_index:rand_index+self.length]
        return obs, action

def train_model(model, device, criterion, optimizer, data_loader, num_epochs, foldername):
    model.to(device)
    for epoch in range(num_epochs):
        for batch_obs, batch_actions in data_loader:
            # Forward pass
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            outputs = model(batch_obs)
            loss = criterion(outputs, batch_actions)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        wandb.log({"loss": loss.item()})
        if (epoch + 1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            # Save the model
        if (epoch + 1) % 100000 == 0:
            torch.save(model.state_dict(), f'{foldername}/{epoch+1}.pth')

    print("Training complete.")
    return model

def train(dataset_paths, save_model_path):
    
    obs = []
    actions=[]
    reset=[]
 
    for dataset_path in dataset_paths:
        obs1, actions1, reset1 = joblib.load(dataset_path)
    
        obs+=obs1
        actions+=actions1
        reset+=reset1
    input_size = obs[0].shape[1]  # Example input size (e.g., pose parameters + shape parameters + global orientation)
    hidden_size = 2048  # Example hidden layer size
    output_size = actions[0].shape[1]  # Example output size (e.g., joint angles)
    batch_size = 16384
    num_epochs = 2000000
    learning_rate = 2e-5
    min_episode_length = 1

    run = wandb.init(
        # Set the project where this run will be logged
        project="train_bc",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "min_episode_length": min_episode_length,
            "input_size": input_size,
            "dataset": dataset_path,
        },
    )

    if len(obs)>11313:
        obs = obs[:11313]
        actions = actions[:11313]
        reset = reset[:11313]
    all_obs, all_actions = [], []

    for index, value in enumerate(reset):
        if value[-1] != 1:
            value[-1] = 1
        curr_index_reset = np.where(value == 1)[0]
        prev_index = 0
        for curr_index in curr_index_reset:
            if curr_index - prev_index + 1 >= min_episode_length:
                all_obs.append(obs[index][prev_index: curr_index + 1])
                all_actions.append(actions[index][prev_index: curr_index + 1])
                prev_index = curr_index + 1


    assert len(all_obs) == len(all_actions)
    all_obs = [torch.tensor(array, dtype=torch.float32).to(device) for array in all_obs]
    all_actions = [torch.tensor(array, dtype=torch.float32).to(device) for array in all_actions]
    dataset = HumanoidDataset(all_obs, all_actions, min_episode_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model, define the loss function and the optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, device, criterion, optimizer, data_loader, num_epochs, save_model_path)


if __name__ == "__main__":
    parent_folder = "./bc_model/obs_clean_actions_reset_action_noise_0.05/"
    pkl_files = [parent_folder+f for f in os.listdir(parent_folder) if f.endswith('.pkl')]
    print(pkl_files) 
    file_name = "first128"
    foldername = os.path.join(parent_folder, file_name)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
 
    train(pkl_files, foldername)

