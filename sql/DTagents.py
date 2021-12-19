from sql.RLagents import Agent
from torch.optim import AdamW
from sql.models.models import *
import torch
import numpy as np
# from decision_transformer.evaluation.evaluate_episodes import *

class DecisionTransformerAgent(Agent):
    def __init__(self, env, gamma=1, hidden_dim=128, lr=0.0001, alpha=1, act_f='relu', n_layer=3,
                 n_head=1, sequence_length=20, weight_decay=0.0001, warmup_steps=100000, warmup_method=1, scale=1000,
                 target_return=3600, device='cpu', grad_norm_clip=0.25, state_mean=0.0, state_std=1.0, max_ep_len=1000):
        Agent.__init__(self, env)

#         print('id', self.state_dim)
        # create model
        self.model = DecisionTransformer(hidden_dim, 5000, self.state_dim, self.action_dim, act_f, n_layer, n_head)

        # attach model to optim. Paper mentions they used AdamW with LR of 0.001
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
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
        pass

    def offline_train(self, states, actions, rewards, dones, rtg, timesteps, mask, per_batch=False):
        """
        Input: tensors of states of dimension Z * K * X, where Z is number of trajectories (batch size), K is trajectory
                length, and X is the dimension of the states
                similar for rewards, ... etc

                Note that the rtgs inputs may have trajectory lengths +1 for padding
                The rtgs are normalized already

        Trains the agent
        """
        self.model.train()
        if per_batch:
            a_preds = self.model(rtg[:, :-1], states, actions, timesteps, mask)

            # this loss is weird -- why r we taking diff in actions?
            loss = torch.mean((a_preds - actions) ** 2)
#             print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
#                 print('sss')
        else:
            # note that we optimize over each batch instead of each data here. the paper optimized per data
            # loop over each batch of size K
            for i in range(len(states)):
                # print(len(states[i]))
                # print(len(rtg[i]))
                # print(len(actions[i]))
                # print(len(timesteps[i]))

                a_preds = self.model(rtg[i][:-1], states[i], actions[i], timesteps[i], mask[i])

#                 print('ap', a_preds)
#                 print('aa', actions[i])

                # # this loss is weird -- why r we taking diff in actions?
                loss = torch.mean((a_preds - actions[i]) ** 2)
#                 print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()


        pass

    def online_evaluate(self, episodes=100, normalized=False):
        
        self.model.eval()
        with torch.no_grad():

            r_per_eps = np.zeros(episodes)
            l_per_eps = np.zeros(episodes)

            for i in range(episodes):
#                 print('eps', i)
                # we want to input tensors here
                R = torch.tensor([[self.target_return/self.scale]], dtype=self.dtype, device=self.device)
                s = torch.tensor([self.env.reset()], dtype=self.dtype, device=self.device)
                a = torch.zeros((0, self.action_dim), dtype=self.dtype, device=self.device)
                t = torch.tensor([1], dtype=self.itype, device=self.device)
                sum_r = 0
                done = False
                length = 0
                while not done:
                    # sample next action. pad action with zeros for next action?
    #                 print('R', R)
    #                 print('A', torch.cat((a, torch.zeros((1, self.action_dim), device=self.device))))
                    action = self.model(R, s, torch.cat((a, torch.zeros((1, self.action_dim), device=self.device))), t)[-1]
                    # print('t sh', t.shape)
                    # print('R sh', R.shape)
                    s_prime, r, done, _ = self.env.step(action.detach().numpy())

                    # append new tokens
                    R = torch.cat((R, torch.tensor([R[-1] - r/self.scale], device=self.device).reshape(1, -1)))
                    s = torch.cat((s, torch.tensor((s_prime- self.state_mean) / self.state_std, dtype=self.dtype, device=self.device).reshape(1, -1)))
                    a = torch.cat((a, torch.tensor(action, dtype=self.dtype, device=self.device).reshape(1, -1)))
                    t = torch.cat((t, torch.tensor([R.shape[0]], device=self.device)))
                    sum_r += r
                    length += 1

                    # print(s.shape)

                    # slice out extra tokens
                    R, s, a, t = R[-self.sequence_length:], s[-self.sequence_length:], a[-self.sequence_length + 1:], t[
                                                                                                                      -self.sequence_length:]
                if normalized:
                    r_per_eps[i] = self.env.get_normalized_score(sum_r)
                else:
                    r_per_eps[i] = sum_r

                l_per_eps[i] = length
            #         , R, s, a, t
        return r_per_eps, l_per_eps
    
    


