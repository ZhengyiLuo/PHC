from uhc.khrylib.utils.torch import *
from torch import nn
from uhc.khrylib.models.rnn import RNN
from uhc.khrylib.models.mlp import MLP


class ERDNet(nn.Module):

    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.encoder_mlp = MLP(state_dim, (500,), 'relu')
        self.encoder_linear = nn.Linear(500, 500)
        self.lstm1 = RNN(500, 1000, 'lstm')
        self.lstm2 = RNN(1000, 1000, 'lstm')
        self.decoder_mlp = MLP(1000, (500, 100), 'relu')
        self.decoder_linear = nn.Linear(100, state_dim)
        self.mode = 'batch'

    def initialize(self, mode):
        self.mode = mode
        self.lstm1.set_mode(mode)
        self.lstm2.set_mode(mode)
        self.lstm1.initialize()
        self.lstm2.initialize()

    def forward(self, x):
        if self.mode == 'batch':
            batch_size = x.shape[1]
            x = x.view(-1, x.shape[-1])
        x = self.encoder_mlp(x)
        x = self.encoder_linear(x)
        if self.mode == 'batch':
            x = x.view(-1, batch_size, x.shape[-1])
        x = self.lstm1(x)
        x = self.lstm2(x)
        if self.mode == 'batch':
            x = x.view(-1, x.shape[-1])
        x = self.decoder_mlp(x)
        x = self.decoder_linear(x)
        return x


if __name__ == '__main__':
    net = ERDNet(64)
    input = ones(32, 3, 64)
    out = net(input)
    print(out.shape)
