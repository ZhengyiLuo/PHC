import torch.nn as nn
from uhc.utils.math_utils import *
from uhc.khrylib.rl.core.distributions import Categorical
from uhc.khrylib.rl.core.policy import Policy


class PolicyDiscrete(Policy):
    def __init__(self, net, action_num, net_out_dim=None):
        super().__init__()
        self.type = 'discrete'
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.net = net
        self.action_head = nn.Linear(net_out_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.net(x)
        action_prob = torch.softmax(self.action_head(x), dim=1)
        return Categorical(probs=action_prob)

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}

