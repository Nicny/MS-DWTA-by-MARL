import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    # related to the Mozi Platform
    parser.add_argument('--avail_ip_port', type=str, default='127.0.0.1:6060')
    parser.add_argument('--side_name', type=str, default='蓝方')
    parser.add_argument('--agent_key_event_file', type=str, default=None)
    parser.add_argument('--platform_mode', type=str, default='development')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=str, default='6060')
    parser.add_argument('--appmode', type=int, default=1)
    parser.add_argument('--synchronous', type=bool, default=True)
    parser.add_argument('--duration_interval', type=int, default=30)
    parser.add_argument('--simulate_compression', type=int, default=4, help='推演档位')
    parser.add_argument('--scenario_name', type=str, default='40-1-4.scen', help='想定名称')
    parser.add_argument('--n_enemies', type=int, default=40, help='敌方弹的数量')

    # related to training
    parser.add_argument('--alg', type=str, default='mmd_marl', help='the algorithm to train the agent')
    parser.add_argument('--n_steps', type=int, default=100000, help='total time steps')
    parser.add_argument('--n_episodes', type=int, default=1, help='the number of episodes before once training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_cycle', type=int, default=2000, help='how often to evaluate the model')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy; runner-plt')

    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--evaluate', type=bool, default=False, help='whether to evaluate the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')

    # related to scene
    parser.add_argument('--enemies', type=int, default=40)
    parser.add_argument('--n_agents', type=int, default=7)
    parser.add_argument('--ship_value', type=list, default=[0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.5])
    parser.add_argument('--n_actions', type=int, default=161)
    parser.add_argument('--state_shape', type=int, default=249)
    parser.add_argument('--obs_shape', type=int, default=295)
    parser.add_argument('--episode_limit', type=int, default=16)
    # parser.add_argument('--', type=, default='')
    args = parser.parse_args()
    return args


# arguments of weighted-qmix、 qtran、 qplex
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 256
    args.qmix_hidden_dim = 128
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 1e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.01
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the train steps in one epoch
    args.train_steps = 1

    # experience replay
    args.batch_size = 64
    args.buffer_size = int(2e3)

    # how often to save the model
    args.save_cycle = 1000

    # how often to update the target_net
    args.target_update_cycle = 50

    # QTRAN lambda
    args.lambda_opt = 1
    args.lambda_nopt = 1

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # QPLEX
    args.adv_hypernet_embed = 64
    args.num_kernel = 10
    args.adv_hypernet_layers = 3
    args.adv_hypernet_embed = 64
    args.weighted_head = True
    args.hypernet_embed = 64
    args.is_minus_one = True
    args.double_q = False

    # our algo
    args.agent_own_state_size = 7
    args.n_query_embedding_layer1 = 64
    args.n_query_embedding_layer2 = 32
    args.n_key_embedding_layer1 = 32
    args.n_head_embedding_layer1 = 64
    args.n_head_embedding_layer2 = 4
    args.n_attention_head = 4
    args.n_constrant_value = 32

    args.particle_num = 4
    return args
