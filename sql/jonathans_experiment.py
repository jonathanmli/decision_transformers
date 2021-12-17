import gym
import numpy as np
import torch
# import wandb

import argparse
import pickle
import random
import sys
from sql.DTagents import *
import time


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def prepare_experiment(
        exp_prefix,
        device='cuda',
        env_name='hopper',
        dataset='medium',
        model_type='dt',
        mode='normal',
        K=20,
        pct_traj=1
):
    """
    The upper portion of experiment was taken from the github repo for the paper.
    It's not very conceptually interesting -- all it does is parse the dataset that they have into
    a readable format, and create a function `get_batch` that allows us to sample from the dataset in batches.

    I've made some comments on their code for readability, but otherwise it remains fairly unchanged.
    """

    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_target = 3600  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_target = 6000
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_target = 5000
        scale = 1000.
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_target = 50
        scale = 10.
    else:
        raise NotImplementedError

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    # print('sd', state_dim)
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists

    # note that we are recording cumreturns?
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    # not sure why we need so much normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # print(traj_lens)
    # print(returns)
    # print(type(states))
    # print(len(traj_lens))
    # print(trajectories[0])
    # print(type(trajectories[0]))
    # print(trajectories[0]['terminals'])

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff

            # not explained here why we take an additional rtg
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
#             print('bs rtg', rtg[i].shape)
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

#             print('bs rtg 2', rtg[i].shape)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    # import things moving forward are:
    # traj_lens -- len of each episode
    # num timesteps -- total number of time steps S->A->R
    # states -- states in each episode, stored as ndarray
    # returns -- some of returns for each episode
    # trajectories -- list or episodes/paths
    # path -- dictionary containing ndarrays of 'rewards', 'actions', 'observations'
    # path either contains 'terminals' or 'dones', boolean indicating absorbing state

    # Code above this was taken from the github repo for the paper.
    # All we need is the get_batch function, which provides the data we need to feed into our model

    # bb = get_batch(2)
    # print(bb[-1].shape)
    # print(type(bb))
    # print(bb[-2])
    #
    # s, a, r, d, rtg, timesteps, mask = get_batch(2)
    #
    # dta = DecisionTransformerAgent(env)
    # dta.offline_train(s, a, r, d, rtg, timesteps, mask)

    return get_batch, env, max_ep_len, scale, env_target, state_mean, state_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--per_batch', type=bool, default=True)
    parser.add_argument('--warmup_method', type=int, default=1)

    args = parser.parse_args()
    variant = vars(args)
    # the parsing above was taken from their code to maintain consistency in experiment running
    

    device = variant.get('device', 'cpu')
    K = variant['K']
    pct_traj = variant.get('pct_traj', 1.)
    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    mode = variant.get('mode', 'normal')
    
    # prepare the experiment using the dataset
    new_batch, env, max_ep_len, scale, env_target, state_mean, state_std = prepare_experiment('gym-experiment', device=device, env_name=env_name, 
                                                                  dataset=dataset, model_type=model_type, mode=mode, 
                                                                  K=K, pct_traj=pct_traj)
    
    log_to_wandb = variant.get('log_to_wandb', False)
    hidden_dim = variant['embed_dim']
    lr = variant['learning_rate']
    act_f = variant['activation_function']
    n_layer = variant['n_layer']
    n_head = variant['n_head']
    weight_decay = variant['weight_decay']
    warmup_steps = variant['warmup_steps']
    dropout = variant['dropout']
    warmup_method = variant['warmup_method']
    
    # create a DT agent
    dta = DecisionTransformerAgent(env, hidden_dim=hidden_dim, lr=lr, act_f=act_f, n_layer=n_layer,
                                   n_head=n_head, sequence_length=K, weight_decay=weight_decay, warmup_steps=warmup_steps,
                                   warmup_method=1, scale=scale, target_return=env_target, device=device, 
                                   state_mean=state_mean, state_std=state_std, max_ep_len=max_ep_len)
#     dta = DecisionTransformerAgent(env, scale=scale, target_return=env_target, warmup_steps=100, warmup_method=1, lr=0.001)
    batch_size = variant['batch_size']
    num_steps_per_iter = variant['num_steps_per_iter']
    per_batch = variant['per_batch']
    
    # train DT agent
    start_time = time.time()
    for i in range(num_steps_per_iter):
        s, a, r, d, rtg, timesteps, mask = new_batch(batch_size)
        dta.offline_train(s, a, r, d, rtg, timesteps, mask, per_batch=per_batch)
        print('Batch number', i)
    training_time = time.time()-start_time
    print('Training time: ', training_time)
    print('Training time per batch', training_time/num_steps_per_iter)
    
    
    
    num_eval_episodes = variant['num_eval_episodes']
    
    # evaluate DT agent
    start_time = time.time()
    returns, lengths = dta.online_evaluate(num_eval_episodes)
    print('mean return', returns.mean())
    print('std returns', returns.std())
    print('mean lengths', returns.mean())
    print('std lengths', returns.std())
    testing_time = time.time()-start_time
    print('Testing time: ', testing_time)