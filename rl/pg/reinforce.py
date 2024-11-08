import argparse
import logging
import time
import random
import sys
import os
import signal
import pdb

import numpy as np
import gym
from gym import ObservationWrapper
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F


signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))


class Policy(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(d_in, 16),
            nn.ReLU(True),
            nn.Linear(16, 16),
            nn.ReLU(True),
            nn.Linear(16, 16),
            nn.ReLU(True),
            nn.Linear(16, d_out),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.main(x)
        return x


class Reshape(ObservationWrapper):
    def observation(self, observation):
        return np.expand_dims(observation, 0)


def line(info):
    pieces = [info['mode'].capitalize()]
    for key, fmt in [('episode', '9d'), ('loss avg', '9.2f'), ('reward', '9.2f'), ('reward avg', '9.2f'),
                     ('reward std', '9.2f'), ('reward avg highest', '9.2f'), ('time', '9.2f')]:
        if key in info:
            pieces.append('{}: {:{fmt}}'.format(key.capitalize(), info[key], fmt=fmt))
    return '\t'.join(pieces)


class Reinforce():
    def __init__(self, args):
        self.env = Reshape(gym.make(args.env_name))
        self.discount = args.discount
        self.d_state = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n
        self.solve_threshold = args.solve_threshold

        if args.seed is not None:
            logging.info('Random seed: {}'.format(args.seed))
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            self.env.seed(args.seed)

        self.train_episodes = args.train_episodes
        self.report_freq = args.report_freq
        self.test_freq = args.test_freq
        self.test_episodes = args.test_episodes

        self.policy = Policy(self.d_state, self.n_action)
        logging.info('Policy: {}'.format(self.policy))

        self.learning_rate = args.learning_rate
        self.optimizer = eval('optim.' + args.optimizer)(self.policy.parameters(), lr=self.learning_rate)
        self.reward_scale = args.reward_scale

        self.gpu = args.gpu and torch.cuda.is_available()
        if self.gpu:
            logging.info('Using GPU {}'.format(torch.cuda.current_device()))
            self.policy.cuda()
        global FloatTensor, LongTensor
        FloatTensor = torch.cuda.FloatTensor if self.gpu else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.gpu else torch.LongTensor

        self.train_rewards = []
        self.test_rewards = []
        self.plot_filename = args.plot_filename

    def plot_rewards(self):
        tst = np.asarray(self.test_rewards)
        plt.errorbar(tst[:, 0], tst[:, 1], yerr=tst[:, 2], color='b', linewidth=0.75)
        plt.ylabel('Average reward')
        plt.xlabel('Episode')
        plt.title(self.env.spec.id)
        plt.savefig(self.plot_filename, format='png')

    def generate_episode(self):
        rewards = []
        log_probs = []
        state = self.env.reset()
        while True:
            pred = self.policy(Variable(FloatTensor(state)))
            dist = Categorical(pred)
            action = dist.sample()
            state_next, reward, is_terminal, _ = self.env.step(action.data[0])
            rewards.append(reward)
            log_probs.append(dist.log_prob(action))
            if is_terminal:
                break
            state = state_next
        return rewards, log_probs

    def train(self):
        total_reward = 0
        total_loss = 0
        total_time = 0

        for i in range(1, self.train_episodes + 1):
            begin_time = time.time()

            rewards, log_probs = self.generate_episode()
            reward_batch = FloatTensor(rewards)
            log_prob_batch = torch.cat(log_probs)

            partial_rewards = torch.zeros_like(reward_batch)
            T = reward_batch.size()[0]
            for t in range(T):
                partial_rewards[t] = torch.sum((self.discount ** torch.arange(T - t)) * reward_batch[t:])

            loss = -torch.mean(self.reward_scale * Variable(partial_rewards) * log_prob_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_reward += sum(rewards)
            total_loss += loss.data[0]
            total_time += time.time() - begin_time

            if i % self.report_freq == 0:
                reward_avg = total_reward / self.report_freq
                loss_avg = total_loss / self.report_freq
                self.train_rewards.append((i, reward_avg))
                logging.info(line({'mode': 'train', 'episode': i, 'loss avg': loss_avg,
                                   'reward avg': reward_avg, 'time': total_time}))
                total_reward = 0
                total_loss = 0
                total_time = 0

            if i % self.test_freq == 0:
                rewards = []
                for j in range(self.test_episodes):
                    episode_reward = sum(self.generate_episode()[0])
                    rewards.append(episode_reward)
                reward_avg = np.mean(rewards)
                reward_std = np.std(rewards)
                self.test_rewards.append((i, reward_avg, reward_std))
                logging.info(line({'mode': 'test', 'reward avg': reward_avg, 'reward std': reward_std,
                                   'reward avg highest': max(list(zip(*self.test_rewards))[1])}))
                if self.solve_threshold is not None and reward_avg > self.solve_threshold:
                    logging.info('Environment solved')
                    break

        if self.plot_filename is not None:
            self.plot_rewards()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int)
    parser.add_argument('--log_level', choices=['debug', 'info'], default='info')
    parser.add_argument('--env_name', default='LunarLander-v2')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--solve_threshold', type=float)

    parser.add_argument('--train_episodes', type=int, default=50000)
    parser.add_argument('--report_freq', type=int, default=50)
    parser.add_argument('--test_freq', type=int, default=100)
    parser.add_argument('--test_episodes', type=int, default=20)

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--reward_scale', type=float, default=0.01)
    parser.add_argument('--gpu', action='store_true')

    parser.add_argument('--plot_filename')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
                        level=getattr(logging, args.log_level.upper()))

    logging.info(args)

    reinforce = Reinforce(args)
    reinforce.train()


if __name__ == '__main__':
    main()
