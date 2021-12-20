

import argparse

import sys
from DTagents import *
import time


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
    parser.add_argument('--warmup_steps', type=int, default=1e5)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=1e5)
    parser.add_argument('--device', type=str, default='cuda')
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
    batch_size = variant['batch_size']
    num_steps_per_iter = variant['num_steps_per_iter']
    per_batch = variant['per_batch']
    num_eval_episodes = variant['num_eval_episodes']
    
    # create a DT agent
    dta = DecisionTransformerAgent(env_name=env_name, hidden_dim=hidden_dim, lr=lr, act_f=act_f, n_layer=n_layer,
                                   n_head=n_head, sequence_length=K, weight_decay=weight_decay, warmup_steps=warmup_steps,
                                   warmup_method=1, dataset=dataset, mode=mode, batch_size=batch_size, pct_traj=pct_traj)
   
    
    # train DT agent
    print("Train")
    dta.train(num_steps_per_iter)
    
    # evaluate DT agent
    print("BM Evaluate")
    dta.evaluate(num_eval_episodes)

    print("Evaluate")
    dta.set_evaluater()
    dta.evaluate(num_eval_episodes)