import torch
import os
import datetime
import torch.nn.functional as F
from mozi_ai_sdk.MS_DWTA_open_source.marl.network.base_d_agent import BasicMAC
from mozi_ai_sdk.MS_DWTA_open_source.marl.network.dmix_net import DMixer


class DMIX:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.n_quantiles = 8
        self.n_target_quantiles = 8

        self.eval_rnn = BasicMAC(args)
        self.target_rnn = BasicMAC(args)
        self.eval_qmix_net = DMixer(args)
        self.target_qmix_net = DMixer(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.agent.cuda()
            self.target_rnn.agent.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.result_dir + '/' + args.alg
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/' + 'rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/' + 'rnn_net_params.pkl'
                path_qmix = self.model_dir + '/' + 'dmix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.agent.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        self.target_rnn.agent.load_state_dict(self.eval_rnn.agent.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.agent.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        print('Init alg DMIX')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()

        mac_out, rnd_quantiles, target_mac_out = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        actions_for_quantiles = u.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)
        chosen_action_qvals = torch.gather(mac_out, dim=3, index=actions_for_quantiles).squeeze(3)
        del actions_for_quantiles
        assert chosen_action_qvals.shape == (episode_num, max_episode_len, self.args.n_agents, self.n_quantiles)
        del u

        assert avail_u_next.shape == (episode_num, max_episode_len, self.args.n_agents, self.args.n_actions)
        target_avail_actions = avail_u_next.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_mac_out[target_avail_actions == 0] = -9999999
        avail_actions = avail_u_next.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)

        cur_max_actions = target_mac_out.mean(dim=4).max(dim=3, keepdim=True)[1]
        assert cur_max_actions.shape == (episode_num, max_episode_len, self.args.n_agents, 1)
        cur_max_actions_ = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions_).squeeze(3)
        del target_mac_out
        assert target_max_qvals.shape == (episode_num, max_episode_len, self.args.n_agents, self.n_target_quantiles)

        q_attend_regs = 0
        chosen_action_qvals = self.eval_qmix_net(chosen_action_qvals, s, target=False)
        target_max_qvals = self.target_qmix_net(target_max_qvals, s_next, target=True)
        assert chosen_action_qvals.shape == (episode_num, max_episode_len, 1, self.n_quantiles)
        assert target_max_qvals.shape == (episode_num, max_episode_len, 1, self.n_target_quantiles)

        target_samples = r.unsqueeze(3) + (self.args.gamma * (1 - terminated)).unsqueeze(3) * target_max_qvals
        del target_max_qvals
        del r
        del terminated
        assert target_samples.shape == (episode_num, max_episode_len, 1, self.n_target_quantiles)

        # Quantile Huber loss
        target_samples = target_samples.unsqueeze(3).expand(-1, -1, -1, self.n_quantiles, -1)
        assert target_samples.shape == (episode_num, max_episode_len, 1, self.n_quantiles, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (episode_num, max_episode_len, 1, self.n_quantiles)
        chosen_action_qvals = chosen_action_qvals.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (episode_num, max_episode_len, 1, self.n_quantiles, self.n_target_quantiles)

        # u is the signed distance matrix
        k = target_samples.detach() - chosen_action_qvals
        del target_samples
        del chosen_action_qvals
        assert k.shape == (episode_num, max_episode_len, 1, self.n_quantiles, self.n_target_quantiles)
        assert rnd_quantiles.shape == (episode_num, max_episode_len, 1, self.n_quantiles)
        tau = rnd_quantiles.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert tau.shape == (episode_num, max_episode_len, 1, self.n_quantiles, self.n_target_quantiles)

        # The abs term in quantile huber loss
        abs_weight = torch.abs(tau - k.le(0.).float())
        del tau
        assert abs_weight.shape == (episode_num, max_episode_len, 1, self.n_quantiles, self.n_target_quantiles)
        # Huber loss
        loss = F.smooth_l1_loss(k, torch.zeros(k.shape).cuda(), reduction='none')
        del k
        assert loss.shape == (episode_num, max_episode_len, 1, self.n_quantiles, self.n_target_quantiles)
        # Quantile Huber loss
        loss = (abs_weight * loss).mean(dim=4).sum(dim=3)
        del abs_weight
        assert loss.shape == (episode_num, max_episode_len, 1)
        assert mask.shape == (episode_num, max_episode_len, 1)
        mask = mask.expand_as(loss)

        # 0-out the targets that came from padded data
        loss = loss * mask
        loss = loss.sum() / mask.sum() + q_attend_regs
        assert loss.shape == ()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.agent.load_state_dict(self.eval_rnn.agent.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        mac_out, rnd_quantiles = [], []
        target_mac_out = []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_rnn.hidden_states = self.eval_rnn.hidden_states.cuda()
                self.target_rnn.hidden_states = self.target_rnn.hidden_states.cuda()
            agent_outs, self.eval_rnn.hidden_states, agent_rnd_quantiles = self.eval_rnn.agent(inputs, self.eval_rnn.hidden_states, forward_type="policy")
            assert agent_outs.shape == (episode_num * self.args.n_agents, self.args.n_actions, self.n_quantiles)
            assert agent_rnd_quantiles.shape == (episode_num * 1, self.n_quantiles)
            agent_rnd_quantiles = agent_rnd_quantiles.view(episode_num, 1, self.n_quantiles)
            rnd_quantiles.append(agent_rnd_quantiles)
            agent_outs = agent_outs.view(episode_num, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            mac_out.append(agent_outs)

            target_agent_outs, self.target_rnn.hidden_states, _ = self.target_rnn.agent(inputs_next, self.target_rnn.hidden_states, forward_type="target")
            assert target_agent_outs.shape == (episode_num * self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_agent_outs = target_agent_outs.view(episode_num, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            assert target_agent_outs.shape == (episode_num, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_mac_out.append(target_agent_outs)

        del agent_outs
        del agent_rnd_quantiles
        mac_out = torch.stack(mac_out, dim=1)  # Concat over time
        rnd_quantiles = torch.stack(rnd_quantiles, dim=1)  # Concat over time
        assert mac_out.shape == (episode_num, max_episode_len, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        assert rnd_quantiles.shape == (episode_num, max_episode_len, 1, self.n_quantiles)

        del target_agent_outs
        del _
        target_mac_out = torch.stack(target_mac_out, dim=1)  # Concat across time
        assert target_mac_out.shape == (episode_num, max_episode_len, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)

        return mac_out, rnd_quantiles, target_mac_out

    def init_hidden(self, episode_num):
        self.eval_rnn.init_hidden_(episode_num)
        self.target_rnn.init_hidden_(episode_num)

    def save_model(self, evaluate_num):
        num = str(evaluate_num)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_dmix_net_params.pkl')
        torch.save(self.eval_rnn.agent.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')
