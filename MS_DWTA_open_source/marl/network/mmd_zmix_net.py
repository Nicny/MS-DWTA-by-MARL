import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class MMD_ATT_MixNet(nn.Module):
    def __init__(self, args):
        super(MMD_ATT_MixNet, self).__init__()
        self.args = args

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.agent_own_state_size = args.agent_own_state_size
        self.u_dim = int(np.prod(self.agent_own_state_size))

        self.n_query_embedding_layer1 = args.n_query_embedding_layer1
        self.n_query_embedding_layer2 = args.n_query_embedding_layer2
        self.n_key_embedding_layer1 = args.n_key_embedding_layer1
        self.n_head_embedding_layer1 = args.n_head_embedding_layer1
        self.n_head_embedding_layer2 = args.n_head_embedding_layer2
        self.n_attention_head = args.n_attention_head
        self.n_constrant_value = args.n_constrant_value

        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(
                nn.Sequential(nn.Linear(self.state_dim - self.agent_own_state_size * self.n_agents,
                                        self.n_query_embedding_layer1), nn.ReLU(),
                              nn.Linear(self.n_query_embedding_layer1,
                                        self.n_query_embedding_layer2)))

        self.key_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.key_embedding_layers.append(nn.Linear(self.u_dim + 3, self.n_key_embedding_layer1))

        self.scaled_product_value = np.sqrt(self.n_query_embedding_layer2)

        self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim * self.args.particle_num)
        self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim),
                                      nn.ReLU(), nn.Linear(args.qmix_hidden_dim, self.args.particle_num))

    def forward(self, q_values, states):
        bs = q_values.size(0)
        states = states.reshape(-1, self.state_dim)  # [bs, max-len, state-dim] --> [bs * max-len, state-dim]
        us, us_m = self._get_us(states)  # us: [bs * max-len * N, agent_feature];  us_m: [bs * max-len, missile_feature]

        q_lambda_list = []
        for i in range(self.n_attention_head):
            us_embedding = self.query_embedding_layers[i](us_m)  # [bs * max-len, 32]
            u_embedding = self.key_embedding_layers[i](us)  # [bs * max-len * N, 32]

            # shape: [-1, 1, state_dim]
            us_embedding = us_embedding.reshape(-1, 1, self.n_query_embedding_layer2)  # [bs * max-len, 1, 32]
            # shape: [-1, state_dim, n_agent]
            u_embedding = u_embedding.reshape(-1, self.n_agents, self.n_key_embedding_layer1)  # [bs * max-len, N, 32]
            u_embedding = u_embedding.permute(0, 2, 1)  # [bs * max-len, 32, N]

            # [bs * max-len, 1, 32] * [bs * max-len, 32, N] ---> [bs * max-len, 1, N]
            raw_lambda = torch.matmul(us_embedding, u_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        # shape: [bs * max-len, n_attention_head, N]
        q_lambda_list = torch.stack(q_lambda_list, dim=1).squeeze(-2)

        # shape: [bs * max-len, N, n_attention_head]
        # q_lambda_list = q_lambda_list.permute(0, 2, 1)

        # [bs * max-len, 1, N] * [bs * max-len, N, n_attention_head] --> [bs * max-len, 1, n_attention_head]
        # [bs, max_ep_len, N, 4] --> [bs, max_ep_len, 4, N] --> [bs * max_ep_len, 4, N]
        q_values = q_values.permute(0, 1, 3, 2).view(-1, self.args.particle_num, self.args.n_agents)
        q_values *= q_lambda_list  # [bs * max-len, n_attention_head, N]

        states = states.reshape(-1, self.args.state_shape)  # [bs * max_ep_len, state_dim]

        w1 = torch.abs(self.hyper_w1(states))  # [bs * max_ep_len, N * hidden]
        b1 = self.hyper_b1(states)  # [bs * max_ep_len, 4 * hidden_dim]
        w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # [bs * max_ep_len, N, hidden_dim]
        b1 = b1.view(-1, self.args.particle_num, self.args.qmix_hidden_dim)  # [bs * max_ep_len, 4, hidden_dim]

        w2 = torch.abs(self.hyper_w2(states))  # [bs * max_ep_len, hidden_dim]
        b2 = self.hyper_b2(states)  # [bs * max_ep_len, 4]
        w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # [bs * max_ep_len, hidden_dim, 1]
        b2 = b2.view(-1, self.args.particle_num, 1)  # [bs * max_ep_len, 4, 1]

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # [bs * max_ep_len, 4, hidden_dim]
        q_total = torch.bmm(hidden, w2) + b2  # [bs * max_ep_len, 4, 1]
        q_total = q_total.permute(0, 2, 1)  # [bs * max_ep_len, 4, 1] --> [bs * max_ep_len, 1, 4]
        q_total = q_total.view(bs, -1, self.args.particle_num)  # [bs, max_ep_len, 4]
        return q_total

    def _get_us(self, states):
        agent_own_state_size = self.agent_own_state_size
        with torch.no_grad():
            bs = states.size(0)
            duty = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                [0., 1., 0.], [0., 1., 0.], [0., 0., 1.]]]).repeat(bs, 1, 1).cuda()
            us = states[:, :agent_own_state_size * self.n_agents].reshape(-1, self.n_agents, agent_own_state_size)
            us = torch.cat([duty, us], dim=-1).reshape(-1, agent_own_state_size + 3)
            # us = states[:, :agent_own_state_size * self.n_agents].reshape(-1, agent_own_state_size)
            us_m = states[:, agent_own_state_size * self.n_agents:]
        return us, us_m