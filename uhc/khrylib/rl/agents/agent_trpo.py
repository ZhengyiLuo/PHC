import scipy.optimize
from uhc.khrylib.rl.agents import AgentPG
from uhc.khrylib.utils import *


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = zeros(b.size())
    if b.is_cuda:
        x.to(b.get_device())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).item()

    for stepfrac in [.5**x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


class AgentTRPO(AgentPG):

    def __init__(self, max_kl=1e-2, damping=1e-2, use_fim=True, **kwargs):
        super().__init__(**kwargs)
        self.max_kl = max_kl
        self.damping = damping
        self.use_fim = use_fim

    def update_value(self, states, returns):

        def get_value_loss(flat_params):
            set_flat_params_to(self.value_net, tensor(flat_params))
            for param in self.value_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)
            values_pred = self.value_net(self.trans_value(states))
            value_loss = (values_pred - returns).pow(2).mean()

            # weight decay
            for param in self.value_net.parameters():
                value_loss += param.pow(2).sum() * 1e-3
            value_loss.backward()
            return value_loss.item(), get_flat_grad_from(self.value_net.parameters()).cpu().numpy()

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                                get_flat_params_from(self.value_net).detach().cpu().numpy(),
                                                                maxiter=25)
        set_flat_params_to(self.value_net, tensor(flat_params))

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        ind = exps.nonzero().squeeze(1)
        self.update_value(states, returns)

        with torch.no_grad():
            fixed_log_probs = self.policy_net.get_log_prob(self.trans_policy(states)[ind], actions[ind])
        """define the loss function for TRPO"""

        def get_loss(volatile=False):
            with torch.set_grad_enabled(not volatile):
                log_probs = self.policy_net.get_log_prob(self.trans_policy(states[ind]), actions[ind])
                action_loss = -advantages[ind] * torch.exp(log_probs - fixed_log_probs)
                return action_loss.mean()

        """use fisher information matrix for Hessian*vector"""

        def Fvp_fim(v):
            M, mu, info = self.policy_net.get_fim(self.trans_policy(states)[ind])
            mu = mu.view(-1)
            filter_input_ids = set([info['std_id']]) if self.policy_net.type == 'gaussian' else set()

            t = ones(mu.size(), requires_grad=True)
            mu_t = (mu * t).sum()
            Jt = compute_flat_grad(mu_t, self.policy_net.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
            Jtv = (Jt * v).sum()
            Jv = torch.autograd.grad(Jtv, t)[0]
            MJv = M * Jv.detach()
            mu_MJv = (MJv * mu).sum()
            JTMJv = compute_flat_grad(mu_MJv, self.policy_net.parameters(), filter_input_ids=filter_input_ids).detach()
            JTMJv /= states.shape[0]
            if self.policy_net.type == 'gaussian':
                std_index = info['std_index']
                JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
            return JTMJv + v * self.damping

        """directly compute Hessian*vector from KL"""

        def Fvp_direct(v):
            kl = self.policy_net.get_kl(self.trans_policy(states)[ind])
            kl = kl.mean()

            grads = torch.autograd.grad(kl, self.policy_net.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * v).sum()
            grads = torch.autograd.grad(kl_v, self.policy_net.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).detach()

            return flat_grad_grad_kl + v * self.damping

        Fvp = Fvp_fim if self.use_fim else Fvp_direct

        loss = get_loss()
        grads = torch.autograd.grad(loss, self.policy_net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

        shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
        lm = math.sqrt(self.max_kl / shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)

        prev_params = get_flat_params_from(self.policy_net)
        success, new_params = line_search(self.policy_net, get_loss, prev_params, fullstep, expected_improve)
        set_flat_params_to(self.policy_net, new_params)
