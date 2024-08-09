import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import joblib


def load_model():
    from train import MLP
    model = MLP(574,2048,69)
    model.load_state_dict(torch.load('./bc_model/bc_model_10.pth'))
    obs = torch.zeros(1,574)
    action = model(obs)


class HumanoidDataset(Dataset):
    def __init__(self, obs, actions, length):
        self.obs = obs
        self.actions = actions
        self.length = length
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if self.length > self.obs[idx].shape[0]:
            print("idx is ", idx, self.obs[idx].shape)
            import pdb; pdb.set_trace()
        obs = self.obs[idx][0:self.length]
        action = self.actions[idx][0:self.length]
        return obs, action

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def train_model(model, device, criterion, optimizer, data_loader, num_epochs):
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

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            # Save the model
        if (epoch + 1) % 2000 == 0:
            torch.save(model.state_dict(), f'./bc_model/bc_model_{epoch+1}.pth')


    print("Training complete.")
    return model

def main():
    # Hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    # Load your dataset here (replace with actual data loading)
    obs, actions, reset = joblib.load("output/HumanoidIm/phc_comp_kp_2/obs_actions_reset.pkl")
    input_size = obs[0].shape[1]  # Example input size (e.g., pose parameters + shape parameters + global orientation)
    hidden_size = 2048  # Example hidden layer size
    output_size = actions[0].shape[1]  # Example output size (e.g., joint angles)
    batch_size = 16384
    num_epochs = 1000000
    learning_rate = 2e-5
    min_episode_length = 16
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
    dataset = HumanoidDataset(all_obs, all_actions, min_episode_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model, define the loss function and the optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, device, criterion, optimizer, data_loader, num_epochs)


if __name__ == "__main__":
    main()
