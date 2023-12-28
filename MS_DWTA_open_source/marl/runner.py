import time
import numpy as np
import os
from rollout import RolloutWorker
from agent import Agents
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, env, args):
        self.env = env
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.idx_1, self.idx_2, self.idx_3, self.idx_4, self.idx_5 = [], [], [], [], []

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def run(self, num):
        time_steps, train_steps, evaluate_steps = 0, 0, 0
        episode_counter = 1

        while time_steps < self.args.n_steps:

            if episode_counter % 500 == 0:
                os.system("taskkill /F /IM MoziServer.exe")
                try:
                    self.env.start()
                except Exception as e:
                    print(f'=====================================================Platform Restart Error：{str(e)}')
                    continue

            try:
                if time_steps % self.args.evaluate_cycle == 0:    # 要被整除才可以；
                    self.evaluate(evaluate_steps)
                    self.plt(0)
                    evaluate_steps += 1

                episodes = []
                # 收集self.args.n_episodes个episodes
                for episode_idx in range(self.args.n_episodes):
                    episode, episode_reward, steps, index_1, index_2, index_3, index_4, index_5 = self.rolloutWorker.generate_episode(episode_idx)
                    print("训练回合为：", episode_counter, "防空效能为：", index_1, "拦截率为：", index_2)
                    episodes.append(episode)
                    episode_counter += 1
                    time_steps += steps

                episode_batch = episodes[0]
                episodes.pop(0)
                for episode in episodes:
                    for key in episode_batch.keys():
                        episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
                if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find(
                        'reinforce') > -1:
                    self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                    train_steps += 1
                else:
                    self.buffer.store_episode(episode_batch)
                    for train_step in range(self.args.train_steps):  # args.train_steps = 1
                        mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                        self.agents.train(mini_batch, train_steps)
                        train_steps += 1

            except Exception as e:
                print(f'============================================ Some place error：{str(e)}')
                while True:
                    try:
                        os.system("taskkill /F /IM MoziServer.exe")
                        time.sleep(1)
                        self.env.start()
                        break
                    except Exception as e:
                        print(f'============================================ Some place error：{str(e)}')
                        continue

        self.evaluate(evaluate_steps)
        self.plt(0)

    def evaluate(self, evaluate_steps):
        print("+++++++++++++++++++++  start to evaluate ! +++++++++++++++++++++")
        average_index_1, average_index_2, average_index_3, average_index_4, average_index_5 = [], [], [], [], []
        ep_reward = None
        for epoch in range(self.args.evaluate_epoch):
            _, ep_reward, _, index_1, index_2, index_3, index_4, index_5 = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            print("====== 评估回合为：", epoch, "防空效能为：", index_1, "拦截率为：", index_2, "高拦截率为：", index_3,
                  "拦截费效为：", index_4, "拦截距离为：", index_5)
            average_index_1.append(index_1)
            average_index_2.append(index_2)
            average_index_3.append(index_3)
            average_index_4.append(index_4)
            average_index_5.append(index_5)
        average_index_1 = sum(average_index_1) / self.args.evaluate_epoch
        average_index_2 = sum(average_index_2) / self.args.evaluate_epoch
        average_index_3 = sum(average_index_3) / self.args.evaluate_epoch
        average_index_4 = sum(average_index_4) / self.args.evaluate_epoch
        average_index_5 = sum(average_index_5) / self.args.evaluate_epoch
        print("防空效能为：", average_index_1, "拦截率为：", average_index_2, "高拦截率为：", average_index_3,
              "拦截费效为：", average_index_4, "拦截距离为：", average_index_5)
        self.idx_1.append(average_index_1)
        self.idx_2.append(average_index_2)
        self.idx_3.append(average_index_3)
        self.idx_4.append(average_index_4)
        self.idx_5.append(average_index_5)
        self.episode_rewards.append(ep_reward)
        # 存储网络模型
        if not self.args.evaluate:
            self.agents.save_policy_network(evaluate_steps)

    def plt(self, num):
        plt.figure()
        plt.ylim([0, 105])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.idx_2)), self.idx_2)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('interception_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')

        np.save(self.save_path + '/air_defense_efficiency_{}'.format(num), self.idx_1)
        np.save(self.save_path + '/interception_rate_{}'.format(num), self.idx_2)
        np.save(self.save_path + '/high_interception_rate_{}'.format(num), self.idx_3)
        np.save(self.save_path + '/interception_cost_{}'.format(num), self.idx_4)
        np.save(self.save_path + '/interception_dist_{}'.format(num), self.idx_5)

        plt.close()
