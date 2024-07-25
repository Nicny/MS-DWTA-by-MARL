import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dmaq_qatten_weight import Qatten_Weight
from .dmaq_si_weight import DMAQ_SI_Weight


class DPLEXMixer(nn.Module):
    def __init__(self, args):
        super(DPLEXMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.attention_weight = Qatten_Weight(args)

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_quantiles
        self.n_agents = args.n_agents

        # lambda_i
        self.si_weight = DMAQ_SI_Weight(args)

    def calc_v(self, agent_qs):
        bs = agent_qs.shape[0]
        n_agents = self.args.n_agents
        assert agent_qs.shape == (bs, n_agents)

        # agent_qs here is chosen action's Q
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = th.sum(agent_qs, dim=-1)
        # Eq.12 (of QPLEX paper) left component
        return v_tot

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        bs = agent_qs.shape[0]
        n_agents = self.args.n_agents
        assert agent_qs.shape == (bs, n_agents)

        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        # individual A
        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        # lambda_i
        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)

        return adv_tot

    def calc(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.shape[0]
        n_agents = self.args.n_agents
        assert agent_qs.shape == (bs, n_agents)

        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)
            return adv_tot

    def forward(self, agent_qs, states, target, actions=None, max_q_i=None, is_v=False):
        batch_size = agent_qs.shape[0]
        episode_length = agent_qs.shape[1]
        if target:
            n_rnd_quantiles = self.n_target_quantiles
        else:
            n_rnd_quantiles = self.n_quantiles
        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, n_rnd_quantiles)
        q_mixture = agent_qs.sum(dim=2, keepdim=True)
        assert q_mixture.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        q_vals_expected = agent_qs.mean(dim=3, keepdim=True)
        q_vals_sum = q_vals_expected.sum(dim=2, keepdim=True)
        assert q_vals_expected.shape == (batch_size, episode_length, self.n_agents, 1)
        assert q_vals_sum.shape == (batch_size, episode_length, 1, 1)

        # Factorization network
        if actions is not None:
            assert actions.shape == (batch_size, episode_length, self.n_agents, self.n_actions)
        if max_q_i is not None:
            assert max_q_i.shape == (batch_size, episode_length, self.n_agents)
        q_joint_expected, attend_mag_regs, head_entropies = self.forward_qplex(q_vals_expected, states, actions, max_q_i, is_v)
        assert q_joint_expected.shape == (batch_size, episode_length, 1, 1)

        if is_v:
            return q_joint_expected, attend_mag_regs, head_entropies

        # Shape network
        q_vals_sum = q_vals_sum.expand(-1, -1, -1, n_rnd_quantiles)
        q_joint_expected = q_joint_expected.expand(-1, -1, -1, n_rnd_quantiles)
        assert q_vals_sum.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        assert q_joint_expected.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        q_joint = q_mixture - q_vals_sum + q_joint_expected
        assert q_joint.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        return q_joint, attend_mag_regs, head_entropies

    def forward_qplex(self, agent_qs, states, actions=None, max_q_i=None, is_v=False):
        bs = agent_qs.size(0)

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1, 1)

        return v_tot, attend_mag_regs, head_entropies
