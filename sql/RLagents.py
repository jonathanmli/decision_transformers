import random
import gym
import sys
import numpy as np
from collections import deque,namedtuple
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from gym.spaces import Discrete, Box
# from util import *


class Agent:
    '''
    Abstract agent class that interfaces with gym env and interfaces between various components.
    Also provides handy functions for recording and plotting rewards.
    offline_extract(Z, K): returns Z batches of sequences of K length
    '''
    def __init__(self, env, offline_extract=None):
        self.env = env
        # print(env.observation_space.shape)
        self.state_dim = np.prod(env.observation_space.shape)
        self.action_dim = env.action_space.shape[0]
        # self.output_dim = env.action_space.n
        self.all_rewards = []
            # np.empty([])

        self.n_episodes = 0
        self.offline_extract = offline_extract

    def observe(self, state):
        return np.reshape(state, self.input_dim)

    def select_action(self, state):
        """
        Agent selects an action based on current policy
        """
        return 0

    def update(self):
        pass

    def train(self):
        """
        Online training by interacting with self.env
        """
        pass

    def plot_rewards(self, window=50, only_mean=False, label='REINFORCE'):
        if not only_mean:
            plt.plot(self.all_rewards, label=label+' raw')
        plt.plot([np.mean(self.all_rewards[i:i + window]) for i in range(self.n_episodes-window)], label=label+' rolling mean')

    def play_episodes(self, n_episodes=5, render=False):
        for i in range(n_episodes):
            done = False
            s = self.env.reset()
            # generate episode
            while not done:
                action, log_prob = self.select_action(s)
                s_prime, reward, done, _ = self.env.step(action)
                s_prime = self.observe(s_prime)
                s = s_prime
                if render:
                    self.env.render()

    def set_env(self, env):
        self.env = env

    def offline_train(self):
        """
        offline training based on input dataset
        """
        pass

    def evaluate(self, n_eps):
        """
        Online evaluation by interacting with self.env
        """
        pass



class ANNAgent(Agent):
    '''
    Deep RL agent with a model, optimizer, loss, trainer, evaluater, and scheduler attached
    '''
    def __init__(self, env, model, optimizer, loss_fn, trainer, evaluater, scheduler=None):
        Agent.__init__(self, env)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn





