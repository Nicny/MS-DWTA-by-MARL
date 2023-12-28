import numpy as np
import torch


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        mount_use_ep = []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # print(epsilon)

        while not terminated and step < self.episode_limit:
            # time.sleep(0.2)
            self.env.get_missile_list()
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot, q_values = [], [], [], np.array([])

            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id, actions)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action, epsilon)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(int(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            step_use_mount = self.env.do_action(actions)
            for i in range(0):
                self.env.scenario_run()
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            mount_use_ep.append(step_use_mount)
            terminate.append([terminated])
            padded.append([0.])
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # self.env.scenario_run()
        r, episode_reward, intercept_sum, win_tag, mount_use_sum, intercept_list, dist, blue_missile\
            = self.env.reward_battle(mount_use_ep)

        health_list = []
        for agent_id in range(self.n_agents):
            health = self.env.get_health_agent(agent_id)
            health_list.append(health)

        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs, because target_q needs avail_action in training
        avail_actions = []
        last_action = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id, last_action)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon

        missile_num = 40
        intercept_rate = intercept_sum / missile_num    # 指标2：回合拦截率
        intercept_dist = dist / missile_num             # 指标5：拦截距离
        if intercept_rate >= 0.95:
            high_interception = 1                       # 指标3：高拦截率
        else:
            high_interception = 0

        intercept_cost = intercept_sum / (mount_use_sum[0] * 0.03 + mount_use_sum[1] * 0.05 +
                                          mount_use_sum[2] * 0.02 + mount_use_sum[3] * 0.04)    # 指标4：费效比
        defense_efficiency = (health_list[0] + health_list[1] + health_list[2]) * 0.6 + \
                             (health_list[3] + health_list[4] + health_list[5]) * 0.8 + health_list[6] * 1 - \
                             (mount_use_sum[0] * 0.03 + mount_use_sum[1] * 0.05 +
                              mount_use_sum[2] * 0.02 + mount_use_sum[3] * 0.04)    # 指标1：防空效能

        return episode, episode_reward, step, \
            defense_efficiency, intercept_rate, high_interception, intercept_cost, intercept_dist

