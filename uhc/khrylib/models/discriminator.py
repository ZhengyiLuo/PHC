import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, net, net_out_dim=None):
        super().__init__()
        self.net = net
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.logic = nn.Linear(net_out_dim, 1)
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        prob = torch.sigmoid(self.logic(x))
        return prob
