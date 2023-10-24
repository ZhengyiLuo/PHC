import torch.nn as nn
from torch.nn.utils import weight_norm
from uhc.khrylib.utils.torch import *


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout, causal):
        super().__init__()
        padding = (kernel_size - 1) * dilation // (1 if causal else 2)
        modules = []
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        modules.append(self.conv1)
        if causal:
            modules.append(Chomp1d(padding))
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        modules.append(self.conv2)
        if causal:
            modules.append(Chomp1d(padding))
        modules.append(nn.ReLU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*modules)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, causal=False):
        super().__init__()
        assert kernel_size % 2 == 1
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     dropout=dropout, causal=causal)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    tcn = TemporalConvNet(4, [1, 2, 8], kernel_size=3, causal=False)
    input = zeros(3, 4, 80)
    out = tcn(input)
    print(tcn)
    print(out.shape)
