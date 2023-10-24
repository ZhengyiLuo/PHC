import torch.nn as nn
from uhc.khrylib.rl.core.distributions import DiagGaussian
from uhc.khrylib.rl.core.policy import Policy
from uhc.utils.math_utils import *
from uhc.khrylib.models.mlp import MLP


class PolicyGaussian(Policy):
    def __init__(self, cfg, action_dim, state_dim, net_out_dim=None):
        super().__init__()
        self.type = "gaussian"
        policy_hsize = cfg.policy_hsize
        policy_htype = cfg.policy_htype
        fix_std = cfg.fix_std
        log_std = cfg.log_std
        self.net = net = MLP(state_dim, policy_hsize, policy_htype)
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(
            torch.ones(1, action_dim) * log_std, requires_grad=not fix_std
        )

    def forward(self, x):
        x = self.net(x)
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return DiagGaussian(action_mean, action_std)

    def get_fim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {"std_id": std_id, "std_index": std_index}
