import torch.nn as nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """This function should return a distribution to sample action from"""
        raise NotImplementedError

    def select_action(self, x, mean_action=False):
        dist = self.forward(x) 
        action = dist.mean_sample() if mean_action else dist.sample()
        return action

    def get_kl(self, x):
        dist = self.forward(x)
        return dist.kl()

    def get_log_prob(self, x, action):
        dist = self.forward(x)
        return dist.log_prob(action)
