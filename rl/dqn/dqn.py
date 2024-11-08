import argparse
import logging
from collections import deque, namedtuple
import time
import random
import sys
import signal
import copy
import re
import pdb
import math

import numpy as np
import gym
from gym import Wrapper
from gym import ObservationWrapper
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


torch.backends.cudnn.deterministic = True


signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))


class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        x = self.fc(x)
        return x


def fc_list(d_in, n_layer, n_hidden, activation):
    modules = nn.ModuleList([nn.Linear(d_in, n_hidden), activation])
    for i in range(n_layer - 1):
        modules.extend([nn.Linear(n_hidden, n_hidden), activation])
    return modules


class MLP(nn.Module):
    def __init__(self, d_in, d_out, n_mlp_layer, n_hidden, activation):
        super().__init__()
        self.main = nn.Sequential(*fc_list(d_in, n_mlp_layer, n_hidden, activation), nn.Linear(n_hidden, d_out))

    def forward(self, x):
        x = self.main(x)
        return x


class DuelingMLP(nn.Module):
    def __init__(self, d_in, d_out, n_mlp_layer, n_stream_layer, n_hidden, activation):
        super().__init__()
        self.main = nn.Sequential(*fc_list(d_in, n_mlp_layer, n_hidden, activation))
        self.advantage = nn.Sequential(
            *fc_list(n_hidden, n_stream_layer, n_hidden // 2, activation),
            nn.Linear(n_hidden // 2, d_out)
        )
        self.value = nn.Sequential(
            *fc_list(n_hidden, n_stream_layer, n_hidden // 2, activation),
            nn.Linear(n_hidden // 2, 1)
        )

    def forward(self, x):
        x = self.main(x)
        adv = self.advantage(x)
        val = self.value(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


class Atari(nn.Module):
    def __init__(self, c_in, d_out, n_hidden):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, n_hidden),
            nn.ReLU(True)
        )
        self.output = nn.Linear(n_hidden, d_out)

    def forward(self, x):
        x = x / 255
        x = self.conv(x)
        x = self.fc(x.view(x.size(0), -1))
        x = self.output(x)
        return x


class DuelingAtari(nn.Module):
    def __init__(self, c_in, d_out, n_hidden):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True)
        )
        self.advantage = nn.Sequential(
            nn.Linear(3136, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, d_out)
        )
        self.value = nn.Sequential(
            nn.Linear(3136, n_hidden),
            nn.ReLU(True),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        x = x / 255
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        adv = self.advantage(x)
        val = self.value(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


class Reshape(ObservationWrapper):
    def observation(self, observation):
        return np.expand_dims(observation, 0)


class Preprocess(ObservationWrapper):
    def __init__(self, env, crop, interpolation):
        super().__init__(env)
        self.crop = crop
        self.interpolation = interpolation

    def observation(self, observation):
        return self.process(observation)

    def process(self, state):
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        if self.crop:
            state = cv2.resize(state, (84, 110), interpolation=self.interpolation)[18:102, :]
        else:
            state = cv2.resize(state, (84, 84), interpolation=self.interpolation)
        return state


class Frameskip(Wrapper):
    def __init__(self, env, action_repeat):
        super().__init__(env)
        self.action_repeat = action_repeat
        self.state_buffer = deque(maxlen=2)

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.state_buffer.clear()
        self.state_buffer.append(state)
        return state

    def step(self, action):
        accumulated_reward = 0
        for _ in range(self.action_repeat):
            state, reward, is_terminal, info = self.env.step(action)
            self.state_buffer.append(state)
            accumulated_reward += reward
            if is_terminal:
                break
        return np.maximum(*self.state_buffer), accumulated_reward, is_terminal, info


class History(Wrapper):
    def __init__(self, env, history_length):
        super().__init__(env)
        self.history_length = history_length
        self.state_buffer = deque(maxlen=self.history_length)

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.state_buffer.clear()
        for _ in range(self.history_length):
            self.state_buffer.append(state)
        return np.stack(self.state_buffer, axis=0)

    def step(self, action):
        state, reward, is_terminal, info = self.env.step(action)
        self.state_buffer.append(state)
        return np.stack(self.state_buffer, axis=0), reward, is_terminal, info


class ReplayMemory():
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def insert(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'state_next', 'is_terminal'])


def batch_sample(sample):
    batch = Experience(*zip(*sample))
    return (Variable(FloatTensor(np.concatenate(batch.state, axis=0))),
            Variable(LongTensor(batch.action).view(-1, 1)),
            Variable(FloatTensor(batch.reward).view(-1, 1)),
            Variable(FloatTensor(np.concatenate(batch.state_next, axis=0))),
            Variable(FloatTensor(np.logical_not(batch.is_terminal).astype(np.uint8)).view(-1, 1)))


def line(info):
    pieces = [info['mode'].capitalize()]
    for key, fmt in [('episode', '9d'), ('reward', '9.2f'), ('reward avg', '9.2f'), ('reward std', '9.2f'),
                     ('reward avg highest', '9.2f'), ('epsilon', '9.4f'), ('iter', '9d')]:
        if key in info:
            pieces.append('{}: {:{fmt}}'.format(key.capitalize(), info[key], fmt=fmt))
    return '\t'.join(pieces)


def wrap_atari(env, action_repeat, history_length, crop, interpolation):
    env = Frameskip(env, action_repeat)
    env = Preprocess(env, crop, interpolation)
    env = History(env, history_length)
    return env


class Agent():
    def __init__(self, args):
        env = gym.make(args.env_name)
        self.action_repeat = args.action_repeat
        self.history_length = args.history_length
        self.crop = args.crop
        self.interpolation = eval('cv2.INTER_' + args.interpolation.upper())
        if args.network == 'atari':
            if 'NoFrameskip-v4' not in args.env_name:
                logging.warn('ALE has frameskip > 1 and repeat_action_probability > 0')
            env = wrap_atari(env, self.action_repeat, self.history_length, self.crop, self.interpolation)
            atari_test_env = wrap_atari(gym.make(args.env_name), self.action_repeat, self.history_length, self.crop,
                                        self.interpolation)
        self.env = Reshape(env)
        self.test_env = self.env if args.network != 'atari' else Reshape(atari_test_env)
        self.discount = args.discount
        self.d_state = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n

        if args.seed is not None:
            logging.info('Random seed: {}'.format(args.seed))
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            self.env.seed(args.seed)
            self.test_env.seed(args.seed)

        self.train_episodes = args.train_episodes
        self.report_freq = args.report_freq
        self.epsilon_init = args.epsilon_init
        self.epsilon_final = args.epsilon_final
        self.epsilon_decay_iter = args.epsilon_decay_iter
        self.epsilon_decay_slope = (self.epsilon_final - self.epsilon_init) / self.epsilon_decay_iter
        self.replay_size = args.replay_size
        self.replay_start_size = args.replay_start_size
        if self.replay_size > 0:
            self.replay_memory = ReplayMemory(self.replay_size)
        else:
            self.replay_memory = None
        self.clip_reward = args.clip_reward

        self.n_mlp_layer = args.n_mlp_layer
        self.n_stream_layer = args.n_stream_layer
        self.n_hidden = args.n_hidden
        self.linear_mc = False
        if args.activation == 'relu':
            self.activation = nn.ReLU(True)
        elif args.activation == 'tanh':
            self.activation = nn.Tanh()
        if args.network == 'atari':
            if args.dueling:
                self.network = DuelingAtari(self.history_length, self.n_action, self.n_hidden)
            else:
                self.network = Atari(self.history_length, self.n_action, self.n_hidden)
        elif args.network == 'mlp':
            if args.dueling:
                self.network = DuelingMLP(self.d_state, self.n_action, self.n_mlp_layer,
                                          self.n_stream_layer, self.n_hidden, self.activation)
            else:
                self.network = MLP(self.d_state, self.n_action, self.n_mlp_layer, self.n_hidden, self.activation)
        elif args.network == 'linear':
            self.network = Linear(self.d_state, self.n_action)
            self.linear_mc = 'MountainCar' in self.env.spec.id
            if self.linear_mc:
                logging.info('Using custom terminal state for linear MountainCar')
        logging.info('Network: {}'.format(self.network))
        self.gpu = args.gpu and torch.cuda.is_available()
        if self.gpu:
            logging.info('Using GPU {}'.format(torch.cuda.current_device()))
            self.network.cuda()
        global FloatTensor, LongTensor
        FloatTensor = torch.cuda.FloatTensor if self.gpu else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.gpu else torch.LongTensor
        self.learning_rate = args.learning_rate
        self.optimizer = eval('optim.' + args.optimizer)(self.network.parameters(), lr=self.learning_rate)
        self.batch_size = args.batch_size
        if args.loss == 'mse':
            self.criterion = nn.MSELoss()
        elif args.loss == 'huber':
            self.criterion = nn.SmoothL1Loss()
        self.async_update_freq = args.async_update_freq
        self.target_update_freq = args.target_update_freq
        if self.target_update_freq > 0:
            self.target_network = copy.deepcopy(self.network)
        else:
            self.target_network = self.network
        self.ddqn = args.ddqn and self.target_update_freq > 0
        self.gradient_clip = args.gradient_clip

        self.test_freq = args.test_freq
        self.test_episodes = args.test_episodes
        self.test_epsilon = args.test_epsilon

        self.train_rewards = []
        self.test_rewards = []
        self.plot_filename = args.plot_filename

        self.model_prefix = args.model_prefix
        self.model_checkpoint = 0
        self.checkpoints = np.floor(np.array([0, 1, 2, 3]) * (self.train_episodes - 1) / 3).astype(np.int32) + 1

    def plot_rewards(self):
        trn = np.asarray(self.train_rewards)
        tst = np.asarray(self.test_rewards)
        plt.plot(trn[:, 0], trn[:, 1], 'r', label='train', linewidth=0.75)
        plt.plot(tst[:, 0], tst[:, 1], 'b', label='test', linewidth=0.75)
        plt.ylabel('Average reward')
        plt.xlabel('Episode')
        plt.title(self.env.spec.id)
        plt.legend()
        plt.savefig(self.plot_filename, format='png')

    def select_action(self, pred, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return torch.max(pred, 1)[1].data[0]

    def decay_epsilon(self, iteration):
        epsilon_decayed = self.epsilon_init + iteration * self.epsilon_decay_slope
        return max(epsilon_decayed, self.epsilon_final)

    def populate_memory(self):
        state = self.env.reset()
        for t in range(self.replay_start_size):
            action = self.env.action_space.sample()
            state_next, reward, is_terminal, _ = self.env.step(action)

            if self.clip_reward:
                reward = np.sign(reward)
            experience = Experience(state, action, reward, state_next,
                                    is_terminal if not self.linear_mc else state_next[0][0] > 0.5)
            self.replay_memory.insert(experience)
            if is_terminal:
                state = self.env.reset()
            else:
                state = state_next

    def train(self):
        if self.replay_memory:
            logging.info('Populating the replay memory for {} iterations'.format(self.replay_start_size))
            self.populate_memory()

        total_reward = 0
        total_iterations = 0
        total_time = 0

        for i in range(1, self.train_episodes + 1):
            begin_time = time.time()
            state = self.env.reset()
            episode_reward = 0
            t = 0

            while True:
                pred = self.network(Variable(FloatTensor(state)))
                action = self.select_action(pred, self.decay_epsilon(total_iterations))
                state_next, reward, is_terminal, _ = self.env.step(action)
                episode_reward += reward

                t += 1
                total_iterations += 1

                if self.clip_reward:
                    reward = np.sign(reward)

                if total_iterations % self.async_update_freq == 0:
                    if self.replay_memory:
                        self.replay_memory.insert(
                            Experience(state, action, reward, state_next,
                                       is_terminal if not self.linear_mc else state_next[0][0] > 0.5))
                        state_batch, action_batch, reward_batch, state_next_batch, mask = batch_sample(
                            self.replay_memory.sample(self.batch_size))
                        pred_batch = self.network(state_batch)
                        if self.ddqn:
                            target_batch = reward_batch + mask * self.discount * \
                                           self.target_network(state_next_batch).gather(
                                               1, torch.max(self.network(state_next_batch), 1, keepdim=True)[1])
                        else:
                            target_batch = reward_batch + mask * self.discount * \
                                           self.target_network(state_next_batch).max(1, keepdim=True)[0]
                        loss = self.criterion(pred_batch.gather(1, action_batch), target_batch.detach())
                    else:
                        target = reward + (0 if (is_terminal if not self.linear_mc else state_next[0][0] > 0.5)
                                           else self.discount * \
                                           self.network(Variable(FloatTensor(state_next))).max().data[0])
                        loss = self.criterion(pred[0, action], Variable(FloatTensor([target])))

                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm(self.network.parameters(), self.gradient_clip)
                    self.optimizer.step()

                if self.target_update_freq > 0 and total_iterations % self.target_update_freq == 0:
                    self.target_network = copy.deepcopy(self.network)

                if is_terminal:
                    break

                state = state_next

            total_reward += episode_reward
            total_time += time.time() - begin_time

            if i % self.report_freq == 0:
                reward_avg = total_reward / self.report_freq
                self.train_rewards.append((i, reward_avg))
                logging.info(line({'mode': 'train', 'episode': i, 'reward avg': reward_avg,
                                   'epsilon': self.decay_epsilon(total_iterations), 'iter': total_iterations}))
                total_reward = 0
            else:
                logging.debug(line({'mode': 'train', 'episode': i, 'reward': episode_reward,
                                    'epsilon': self.decay_epsilon(total_iterations), 'iter': total_iterations}))

            if self.model_prefix is not None and i == self.checkpoints[self.model_checkpoint]:
                filename = self.model_prefix + str(self.model_checkpoint) + '.pth'
                torch.save(self.network, filename)
                logging.info('Saved network to file: {}, checkpoint: {}/3'.format(filename, self.model_checkpoint))
                self.model_checkpoint += 1

            if i % self.test_freq == 0:
                reward_avg, reward_std = self.test()
                self.test_rewards.append((i, reward_avg))
                logging.info(line({'mode': 'test', 'reward avg': reward_avg, 'reward std': reward_std,
                                   'reward avg highest': max(list(zip(*self.test_rewards))[1])}))

        if self.plot_filename is not None:
            self.plot_rewards()

    def test(self):
        rewards = []

        for i in range(1, self.test_episodes + 1):
            state = self.test_env.reset()
            episode_reward = 0
            while True:
                pred = self.network(Variable(FloatTensor(state)))
                state, reward, is_terminal, _ = self.test_env.step(self.select_action(pred, self.test_epsilon))
                episode_reward += reward
                if is_terminal:
                    break
            rewards.append(episode_reward)

        return np.mean(rewards), np.std(rewards)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int)
    parser.add_argument('--log_level', choices=['debug', 'info'], default='info')
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--discount', type=float, default=0.99)

    parser.add_argument('--train_episodes', type=int, default=10000)
    parser.add_argument('--report_freq', type=int, default=500)
    parser.add_argument('--epsilon_init', type=float, default=1.0)
    parser.add_argument('--epsilon_final', type=float, default=0.1)
    parser.add_argument('--epsilon_decay_iter', type=int, default=1000000)
    parser.add_argument('--replay_size', type=int, default=50000)
    parser.add_argument('--replay_start_size', type=int, default=10000)
    parser.add_argument('--clip_reward', action='store_true')

    parser.add_argument('--network', choices=['linear', 'mlp', 'atari'], default='mlp')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--n_mlp_layer', type=int, default=2)
    parser.add_argument('--n_stream_layer', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=16)
    parser.add_argument('--activation', choices=['relu', 'tanh'], default='relu')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--async_update_freq', type=int, default=1)
    parser.add_argument('--target_update_freq', type=int, default=10000)
    parser.add_argument('--ddqn', action='store_true')
    parser.add_argument('--dueling', action='store_true')
    parser.add_argument('--loss', choices=['mse', 'huber'], default='mse')
    parser.add_argument('--gradient_clip', type=float, default=0)

    parser.add_argument('--history_length', type=int, default=4)
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--interpolation', default='linear')

    parser.add_argument('--test_freq', type=int, default=1000)
    parser.add_argument('--test_episodes', type=int, default=100)
    parser.add_argument('--test_epsilon', type=float, default=0.05)

    parser.add_argument('--plot_filename')
    parser.add_argument('--model_prefix')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--record_dirname')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S',
                        level=getattr(logging, args.log_level.upper()))

    logging.info(args)

    agent = Agent(args)

    if args.load:
        if args.record_dirname is not None:
            agent.test_env = gym.wrappers.Monitor(agent.test_env, args.record_dirname, force=True,
                                                  video_callable=lambda x: True)
        for k in [0, 1, 2, 3]:
            filename = agent.model_prefix + str(k) + '.pth'
            agent.network = torch.load(filename, map_location=lambda storage, loc: storage)
            logging.info('Loaded network from file: {}'.format(filename))
            reward_avg, reward_std = agent.test()
            logging.info(line({'mode': 'test', 'reward avg': reward_avg, 'reward std': reward_std}))
    else:
        agent.train()


if __name__ == '__main__':
    main()
