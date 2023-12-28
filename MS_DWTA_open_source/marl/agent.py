import numpy as np
import torch


class Agents:
    def __init__(self, args, k=None):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'qtran_base':
            from policy.qtran_base import QtranBase
            self.policy = QtranBase(args)
        elif args.alg == 'qtran_alt':
            from policy.qtran_alt import QtranAlt
            self.policy = QtranAlt(args)
        elif args.alg == 'qplex':
            from policy.qplex import QPLEX
            self.policy = QPLEX(args)
        elif args.alg == 'w_qmix':
            from policy.weighted_qmix import W_QMIX
            self.policy = W_QMIX(args)
        elif args.alg == 'mmd_marl':
            from policy.mmd_marl import MMD_MARL
            self.policy = MMD_MARL(args)
        else:
            raise Exception("No such algorithm")
        self.args = args

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None):
        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).cuda()
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        if self.args.alg == "mmd_marl":
            hidden_state = self.policy.eval_hidden[:, agent_num, :].cuda()
            z_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
            z_value = z_value.view(z_value.size(0), self.args.n_actions, self.args.particle_num)
            q_value = z_value.mean(-1)   # [1, action_dim, 4] ---> [1, action_dim]
        else:
            hidden_state = self.policy.eval_hidden[:, agent_num, :].cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)
        return action

    def choose_value(self, obs, last_action, agent_num, avail_actions):
        inputs = obs.copy()
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).cuda()
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

        hidden_state = self.policy.eval_hidden[:, agent_num, :].cuda()
        q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

        q_value[avail_actions == 0.0] = - float("inf")
        return q_value.squeeze(0)

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training
        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]    # batch的长度从episode_limit缩短到max_episode_len
        self.policy.learn(batch, max_episode_len, train_step, epsilon)

    def save_policy_network(self, evaluate_num):
        self.policy.save_model(evaluate_num)

