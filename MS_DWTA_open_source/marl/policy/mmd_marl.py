import torch
import os
from mozi_ai_sdk.MS_DWTA_open_source.marl.network.base_net import MMD_RNN
from mozi_ai_sdk.MS_DWTA_open_source.marl.network.mmd_zmix_net import MMD_ATT_MixNet


class MMD_MARL:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        if args.last_action:    # True
            input_shape += self.n_actions
        if args.reuse_network:    # True
            input_shape += self.n_agents

        self.eval_rnn = MMD_RNN(input_shape, args)
        self.target_rnn = MMD_RNN(input_shape, args)
        self.eval_qmix_net = MMD_ATT_MixNet(args)
        self.target_qmix_net = MMD_ATT_MixNet(args)
        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
        self.model_dir = args.result_dir + '/' + args.alg
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/' + 'rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/' + 'rnn_net_params.pkl'
                path_qmix = self.model_dir + '/' + 'mix_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qmix))
            else:
                raise Exception("No model!")

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        self.eval_hidden = None
        self.eval_hidden_ = None
        self.target_hidden = None
        print('Init alg MMD-MIX')

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

        z_evals, z_evals_, z_targets = self.get_z_values(batch, max_episode_len)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        z_evals = z_evals.view(z_evals.size(0), z_evals.size(1), self.args.n_agents, self.args.n_actions, self.args.particle_num)
        z_evals_ = z_evals_.view(z_evals_.size(0), z_evals_.size(1), self.args.n_agents, self.args.n_actions, self.args.particle_num)
        z_targets = z_targets.view(z_targets.size(0), z_targets.size(1), self.args.n_agents, self.args.n_actions, self.args.particle_num)

        u = u.unsqueeze(-1).repeat([1, 1, 1, 1, self.args.particle_num])
        z_evals = torch.gather(z_evals, dim=3, index=u).squeeze(3)    # [bs, max_ep_len, N, 4]

        avail_u_next = avail_u_next.unsqueeze(-1).repeat([1, 1, 1, 1, self.args.particle_num])
        z_evals_[avail_u_next == 0.0] = - 9999999
        argmax_u = z_evals_.mean(-1).max(-1)[1].unsqueeze(-1)    # [bs, max_ep_len, N, 1]
        argmax_u = argmax_u.unsqueeze(-1).repeat([1, 1, 1, 1, self.args.particle_num])    # [bs, max_ep_len, N, 1, 4]
        z_targets = torch.gather(z_targets, dim=3, index=argmax_u).squeeze(3)    # [bs, max_ep_len, N, 4]

        z_total_eval = self.eval_qmix_net(z_evals, s)  # [bs, ep_len, 4]
        z_total_target = self.target_qmix_net(z_targets, s_next)  # [bs, ep_len, 4]
        targets = r + self.args.gamma * z_total_target * (1 - terminated)

        z_total_eval = (mask * z_total_eval).view(-1, self.args.particle_num)  # [bs * ep_len, 4]
        targets = (mask * targets).view(-1, self.args.particle_num)  # [bs * ep_len, 4]

        first_item, second_item, third_item = self.calc_kernel(targets, z_total_eval)  # [bs * ep_len, 4, 4]
        first_item = (first_item.sum(-1).sum(-1) / (self.args.particle_num ** 2)).mean()
        second_item = (second_item.sum(-1).sum(-1) / (self.args.particle_num ** 2)).mean()
        third_item = (third_item.sum(-1).sum(-1) / (self.args.particle_num ** 2)).mean()
        loss = first_item + second_item - 2 * third_item

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def calc_kernel(self, target_q_tot, q_tot, kernel_num=20):
        first_kernel = (q_tot.unsqueeze(-1) - q_tot.unsqueeze(-2)).pow(2)  # [bs, 32, 32]
        second_kernel = (target_q_tot.unsqueeze(-1) - target_q_tot.unsqueeze(-2)).pow(2)  # [bs, 32, 32]
        third_kernel = (q_tot.unsqueeze(-1) - target_q_tot.unsqueeze(-2)).pow(2)  # [bs, 32, 32]
        bandwidth_list = [2 ** i for i in range(kernel_num)]  # list, len=20
        first_items, second_items, third_items = 0, 0, 0
        for h in bandwidth_list:
            h = 2 * (h ** 2)
            first_inner_distance = (-first_kernel / h)
            second_inner_distance = (-second_kernel / h)
            intra_distance = (-third_kernel / h)
            first_items += first_inner_distance
            second_items += second_inner_distance
            third_items += intra_distance
        return first_items, second_items, third_items

    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:    # True
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

    def get_z_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        z_evals, z_evals_, z_targets = [], [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.eval_hidden_ = self.eval_hidden_.cuda()
                self.target_hidden = self.target_hidden.cuda()

            z_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            z_eval_, self.eval_hidden_ = self.eval_rnn(inputs_next, self.eval_hidden_)
            z_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            z_eval = z_eval.view(episode_num, self.n_agents, -1)
            z_eval_ = z_eval_.view(episode_num, self.n_agents, -1)
            z_target = z_target.view(episode_num, self.n_agents, -1)

            z_evals.append(z_eval)
            z_evals_.append(z_eval_)
            z_targets.append(z_target)
        z_evals = torch.stack(z_evals, dim=1)
        z_evals_ = torch.stack(z_evals_, dim=1)
        z_targets = torch.stack(z_targets, dim=1)
        return z_evals, z_evals_, z_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_hidden_ = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, evaluate_num):
        num = str(evaluate_num)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qmix_net.state_dict(), self.model_dir + '/' + num + '_mix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')
