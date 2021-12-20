from RLagents import Agent
from torch.optim import AdamW
from models.models import *
import torch
import numpy as np
from evaluation.DTevaluation import DT_Evaluater, BM_Evaluater
from training.DTtrainer import DT_Trainer
from env.dataset_prepare import prepare_experiment
# from decision_transformer.evaluation.evaluate_episodes import *

class DecisionTransformerAgent(Agent):
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
            scale=1000,
            target_return=3600, 
            grad_norm_clip=0.25, 
            state_mean=0.0, 
            state_std=1.0, 
            max_ep_len=1000, 
            batch_size=256,
            env_name='hopper', 
            dataset='medium',
            mode='normal', 
            pct_traj=1
            ):
        '''
        All experimental parameters should be arguments agent here
        '''
        # check if cuda
        self.on_cuda = torch.cuda.is_available()
        if self.on_cuda:
            device='cuda'
        else:
            device='cpu'

        # get variables from dataset
        batch_sampler, env, max_ep_len, scale, env_target, state_mean, state_std = prepare_experiment('gym-experiment', device=device, env_name=env_name, dataset=dataset, mode=mode, K=sequence_length, pct_traj=pct_traj)

        Agent.__init__(self, env)
        
        # attach model
        self.model = DecisionTransformer(hidden_dim, 5000, self.state_dim, self.action_dim, act_f, n_layer, n_head, device=device, sequence_length=sequence_length)
        if self.on_cuda:
            self.model=self.model.cuda()

        # attach optimizer. Paper mentions they used AdamW with LR of 0.001
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # attach scheduler
        self.scheduler = None
        if warmup_method == 1:
            # linear warmup -- not quite sure why this helps. in fact, I think I see better results without warmup
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                               lr_lambda=lambda x: min(1, (x + 1) / warmup_steps))
        elif warmup_method == 2:
            # we might get better results if we used warmup from attention is all you need paper
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                               lr_lambda=lambda x: warmup_steps**(0.5) * min((x+1)**(-0.5), (x + 1) * (warmup_steps)**(-1.5)))
        
        # some parameters
        self.sequence_length = sequence_length
        self.dtype = torch.float
        self.itype = torch.int
        self.scale = scale
        self.target_return = target_return
        self.device = device
        self.grad_norm_clip = grad_norm_clip
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_ep_len = max_ep_len
        self.mode = mode

        # attach sampler
        self.batch_sampler = batch_sampler

        # attach trainer
        self.trainer = DT_Trainer(self.model, self.optimizer, self.scheduler, batch_sampler, batch_size=batch_size, grad_norm_clip=grad_norm_clip)

        # attach evaluater
        self.set_evaluater(evaluater=BM_Evaluater)
 

    def train(self, n_batches):
        self.trainer.train(n_batches)

    def evaluate(self, episodes=100, normalized=False, **kwargs):
        self.evaluater.evaluate(self.env, self.model)
        self.evaluater.print_summary()
    
    def set_evaluater(self, evaluater=DT_Evaluater):
        self.evaluater = evaluater(dtype=self.dtype, itype=self.itype, sequence_length=self.sequence_length, target_return=self.target_return, max_ep_len=self.max_ep_len, scale=self.scale, device=self.device, state_mean=self.state_mean, state_std=self.state_std, mode=self.mode)
        
    
    


