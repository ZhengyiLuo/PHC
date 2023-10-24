import torch.nn as nn
import torch


class CMLP(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dims=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.cond_dim = cond_dim
        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(nn.Linear(last_dim + cond_dim, nh))
            last_dim = nh

    def forward(self, c, x):
        for affine in self.affine_layers:
            x = torch.cat((c, x), dim=1)
            x = self.activation(affine(x))
        return x
