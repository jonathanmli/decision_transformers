from sql.RLagents import Agent
from torch.optim import AdamW
from sql.models.models import *
import torch
import numpy as np
from sql.evaluation.DTevaluation import DT_Evaluater, BM_Evaluater
from sql.training.DTtrainer import DT_Trainer
# from decision_transformer.evaluation.evaluate_episodes import *

class DecisionTransformerAgent(Agent):
    def __init__(self, env, gamma=1, hidden_dim=128, lr=0.0001, alpha=1, act_f='relu', n_layer=3,
                 n_head=1, sequence_length=20, weight_decay=0.0001, warmup_steps=100000, warmup_method=1, scale=1000,
                 target_return=3600, device='cpu', grad_norm_clip=0.25, state_mean=0.0, state_std=1.0, max_ep_len=1000):
        Agent.__init__(self, env)

        # attach model
        self.model = DecisionTransformer(hidden_dim, 5000, self.state_dim, self.action_dim, act_f, n_layer, n_head)

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

        # attach trainer
        self.trainer = DT_Trainer(self.model, None)

        # attach evaluater
        self.evaluater = BM_Evaluater()
 

    def train(self, n_batches):
        self.trainer.train(n_batches)
        pass

    def evaluate(self, episodes=100, normalized=False, **kwargs):
        self.evaluater.evaluate(self.model)
        self.evaluater.print_summary()
        
    
    


