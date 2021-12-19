import torch
import numpy as np
import matplotlib.pyplot as plt


class Evaluater:
    '''
    Evaluates models and records results of evaluations.
    Also provides methods to visualize and summarize such evaluations
    '''
    def __init__(self):
        self.all_rewards = np.zeros(0)
        self.all_lengths = np.zeros(0)
        pass

    def evaluate(self, env, model, n_eps = 100):
        '''
        Evaluates model against env
        Returns np array of returns and episode lengths
        '''
        pass

    def record(self, rewards, lengths):
        self.all_lengths = np.concatenate((self.all_lengths, lengths))
        self.all_rewards = np.concatenate((self.all_rewards, rewards))

    def print_summary(self):
        print('mean return', self.all_rewards.mean())
        print('std returns', self.all_rewards.std())
        print('mean lengths', self.all_lengths.mean())
        print('std lengths', self.all_lengths.std())

    def plot_rewards(self, window=50, only_mean=False, label='rewards'):
        if not only_mean:
            plt.plot(self.all_rewards, label=label+' raw')
        plt.plot([np.mean(self.all_rewards[i:i + window]) for i in range(self.n_episodes-window)], label=label+' rolling mean')

