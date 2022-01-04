import torch
import numpy as np
import matplotlib.pyplot as plt
import time


class Evaluater:
    '''
    Evaluates models and records results of evaluations.
    Also provides methods to visualize and summarize such evaluations
    '''
    def __init__(self):
        self.all_rewards = np.zeros(0)
        self.all_lengths = np.zeros(0)
        pass

    def evaluate(self, env, model, **kwargs):
        '''
        Evaluates model against env
        Returns np array of returns and episode lengths
        '''
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            self._evaluate(env, model, **kwargs)
        print('evaluation time: ', time.time() - start_time)
        pass

    def _evaluate(self, env, model):
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

