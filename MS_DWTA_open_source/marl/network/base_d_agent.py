import numpy as np
import torch
from torch.distributions import Categorical
from mozi_ai_sdk.MS_DWTA_open_source.marl.network.base_net_d import IQNRNNAgent


class BasicMAC:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.args = args

        input_shape = self.args.obs_shape
        if args.last_action:
            input_shape += self.args.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.agent = IQNRNNAgent(input_shape, args)
        self.action_selector = EpsilonGreedyActionSelector(args)

        self.hidden_states = None

    def init_hidden_(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def choose_action_(self, inputs, avail_actions, epsilon, agent_num):
        # Only select actions for the selected batch elements in bs
        agent_outputs, self.hidden_states[:, agent_num, :], rnd_quantiles = self.agent(inputs, self.hidden_states[:, agent_num, :], forward_type="approx")
        agent_outputs = agent_outputs.mean(dim=-1)    # [bs, action_dim, quant]
        chosen_actions = self.action_selector.select_action(agent_outputs, avail_actions, epsilon)
        return chosen_actions


class EpsilonGreedyActionSelector():
    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, epsilon):
        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected

        random_numbers = torch.rand_like(agent_inputs[:, 0]).cuda()
        pick_random = (random_numbers < epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long().cuda()

        # a = masked_q_values.max(dim=-1)[1]
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=-1)[1]
        return picked_actions
