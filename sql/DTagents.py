from RLagents import Agent, Configuration
from torch.optim import AdamW
from models.models import *
import torch
import numpy as np
from evaluation.DTevaluation import DT_Evaluater, BM_Evaluater
from training.DTtrainer import DT_Trainer
from env.dataset_prepare import prepare_experiment

# from decision_transformer.evaluation.evaluate_episodes import *

class DTConfiguration(Configuration):
    '''
    Sets default parameters for a DT agent
    '''
    def __init__(self, 
            gamma=1, 
            hidden_dim=128, 
            lr=0.0001, 
            alpha=1, 
            act_f='relu', 
            n_layer=3,
            n_head=1, 
            sequence_length=20, 
            weight_decay=0.0001, 
            warmup_steps=100000, 
            warmup_method=1, 
            grad_norm_clip=0.25, 
            batch_size=256,
            env_name='hopper', 
            dataset='medium',
            mode='normal', 
            pct_traj=1,
            seed = 6, 
            training_steps = 10000, 
            n_eval_episodes = 100) -> None:
        super().__init__(
            gamma=gamma, 
            hidden_dim=hidden_dim, 
            lr=lr, 
            alpha=alpha, 
            act_f=act_f, 
            n_layer=n_layer, 
            n_head=n_head, 
            sequence_length=sequence_length, 
            weight_decay=weight_decay, 
            warmup_method=warmup_method, 
            warmup_steps=warmup_steps, 
            grad_norm_clip=grad_norm_clip, 
            batch_size=batch_size, 
            env_name=env_name, 
            dataset=dataset, 
            mode=mode, 
            pct_traj=pct_traj, 
            seed=seed, 
            n_eval_episodes=n_eval_episodes, 
            training_steps=training_steps)

class DecisionTransformerAgent(Agent):
    def __init__(self, config):
        '''
        All experimental parameters should be in config
        '''

        self.config = config
        
        # check if cuda
        self.on_cuda = torch.cuda.is_available()
        if self.on_cuda:
            device='cuda'
        else:
            device='cpu'

        # get variables from dataset
        batch_sampler, env, max_ep_len, scale, env_target, state_mean, state_std = prepare_experiment('gym-experiment', device=device, env_name=config.env_name, dataset=config.dataset, mode=config.mode, K=config.sequence_length, pct_traj=config.pct_traj)
        self.config.max_ep_len = max_ep_len
        self.config.scale = scale
        self.config.env_target = env_target
        # self.config.state_mean = state_mean
        # self.config.state_std = state_std


        Agent.__init__(self, env)
        
        # attach model
        self.model = DecisionTransformer(config.hidden_dim, 5000, 
            self.state_dim, 
            self.action_dim, 
            config.act_f, 
            config.n_layer, 
            config.n_head, 
            device=device, 
            sequence_length=config.sequence_length)
        if self.on_cuda:
            self.model=self.model.cuda()

        # attach optimizer. Paper mentions they used AdamW with LR of 0.001
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # attach scheduler
        self.scheduler = None
        if config.warmup_method == 1:
            # linear warmup -- not quite sure why this helps. in fact, I think I see better results without warmup
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                               lr_lambda=lambda x: min(1, (x + 1) / config.warmup_steps))
        elif config.warmup_method == 2:
            # we might get better results if we used warmup from attention is all you need paper
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                               lr_lambda=lambda x: config.warmup_steps**(0.5) * min((x+1)**(-0.5), (x + 1) * (config.warmup_steps)**(-1.5)))
        
        # some parameters
        self.dtype = torch.float
        self.itype = torch.int

        # the following should all be in config
        self.sequence_length = config.sequence_length
        self.scale = scale
        self.target_return = env_target
        self.device = device
        self.grad_norm_clip = config.grad_norm_clip
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_ep_len = max_ep_len
        self.mode = config.mode
        self.env_name = config.env_name
        self.dataset = config.dataset

        # attach sampler
        self.batch_sampler = batch_sampler

        # attach trainer
        self.trainer = DT_Trainer(self.model, self.optimizer, self.scheduler, batch_sampler, batch_size=config.batch_size, grad_norm_clip=config.grad_norm_clip)

        # attach evaluater
        self.set_evaluater(evaluater=BM_Evaluater)
 

    def train(self, n_batches):
        self.trainer.train(n_batches)

    def evaluate(self, episodes=100, normalized=False, **kwargs):
        self.evaluater.evaluate(self.env, self.model, n_eps=episodes)
        self.evaluater.print_summary()
        # print relevant stats here
        printdict = self.config.__dict__
        printdict['return_mean'] = self.evaluater.all_rewards.mean()
        printdict['return_std'] = self.evaluater.all_rewards.std()
        printdict['length_mean'] = self.evaluater.all_lengths.mean()
        printdict['length_std'] = self.evaluater.all_lengths.std()
        printdict['model'] = 'BM'
        printdict['evaluater'] = str(self.evaluater)
        print(printdict)
        # print(f'{self.evaluater.all_rewards.mean()},{self.evaluater.all_rewards.std()},{self.evaluater.all_lengths.mean()},{self.evaluater.all_lengths.std()},{self.seed}')
    
    def set_evaluater(self, evaluater=DT_Evaluater):
        self.evaluater = evaluater(dtype=self.dtype, itype=self.itype, sequence_length=self.sequence_length, target_return=self.target_return, max_ep_len=self.max_ep_len, scale=self.scale, device=self.device, state_mean=self.state_mean, state_std=self.state_std, mode=self.mode)
        
    
    


