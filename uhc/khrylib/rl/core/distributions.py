import torch
from torch.distributions import Normal
from torch.distributions import Categorical as TorchCategorical


class DiagGaussian(Normal):

    def __init__(self, loc, scale):
        super().__init__(loc, scale)

    def kl(self):
        loc1 = self.loc
        scale1 = self.scale
        log_scale1 = self.scale.log()
        loc0 = self.loc.detach()
        scale0 = self.scale.detach()
        log_scale0 = log_scale1.detach()
        kl = log_scale1 - log_scale0 + (scale0.pow(2) + (loc0 - loc1).pow(2)) / (2.0 * scale1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_prob(self, value):
        return super().log_prob(value).sum(1, keepdim=True)

    def mean_sample(self):
        return self.loc


class Categorical(TorchCategorical):

    def __init__(self, probs=None, logits=None):
        super().__init__(probs, logits)

    def kl(self):
        loc1 = self.loc
        scale1 = self.scale
        log_scale1 = self.scale.log()
        loc0 = self.loc.detach()
        scale0 = self.scale.detach()
        log_scale0 = log_scale1.detach()
        kl = log_scale1 - log_scale0 + (scale0.pow(2) + (loc0 - loc1).pow(2)) / (2.0 * scale1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def log_prob(self, value):
        return super().log_prob(value).unsqueeze(1)

    def mean_sample(self):
        return self.probs.argmax(dim=1)
