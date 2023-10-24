from uhc.khrylib.rl.core import estimate_advantages
from uhc.khrylib.rl.agents.agent import Agent
from uhc.khrylib.utils.torch import *
import time


class AgentPG(Agent):

    def __init__(self, tau=0.95, optimizer_policy=None, optimizer_value=None,
                 opt_num_epochs=1, value_opt_niter=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.opt_num_epochs = opt_num_epochs
        self.value_opt_niter = value_opt_niter

    def update_value(self, states, returns):
        """update critic"""
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(self.trans_value(states))
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        # use a2c by default
        ind = exps.nonzero().squeeze(1)
        for _ in range(self.opt_num_epochs):
            self.update_value(states, returns)
            log_probs = self.policy_net.get_log_prob(self.trans_policy(states)[ind], actions[ind])
            policy_loss = -(log_probs * advantages[ind]).mean()
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

    def update_params(self, batch):
        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(states))

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)

        self.update_policy(states, actions, returns, advantages, exps)

        return time.time() - t0