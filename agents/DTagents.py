from RLagents import Agent
from torch.optim import AdamW
from models import *
import torch
import numpy as np
# from decision_transformer.evaluation.evaluate_episodes import *

class DecisionTransformerAgent(Agent):
    def __init__(self, env, gamma=1, hidden_dim=128, lr=0.0001, alpha=1, act_f='relu', n_layer=3,
                 n_head=1, sequence_length=20, weight_decay=0.0001, warmup_steps=100000, warmup_method=1, scale=100,
                 target_return=3600, device='cpu', grad_norm_clip=0.25, state_mean=0.0, state_std=1.0, max_ep_len=1000):
        Agent.__init__(self, env)

#         print('id', self.input_dim)
        # create model
        self.model = DecisionTransformer(hidden_dim, 5000, self.input_dim, self.output_dim, act_f, n_layer, n_head)

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
                a = torch.zeros((0, self.output_dim), dtype=self.dtype, device=self.device)
                t = torch.tensor([1], dtype=self.itype, device=self.device)
                sum_r = 0
                done = False
                length = 0
                while not done:
                    # sample next action. pad action with zeros for next action?
    #                 print('R', R)
    #                 print('A', torch.cat((a, torch.zeros((1, self.output_dim), device=self.device))))
                    action = self.model(R, s, torch.cat((a, torch.zeros((1, self.output_dim), device=self.device))), t)[-1]
                    s_prime, r, done, _ = self.env.step(action.detach().numpy())

                    # append new tokens
                    R = torch.cat((R, torch.tensor([R[-1] - r/self.scale], device=self.device).reshape(1, -1)))
                    s = torch.cat((s, torch.tensor((s_prime- self.state_mean) / self.state_std, dtype=self.dtype, device=self.device).reshape(1, -1)))
                    a = torch.cat((a, torch.tensor(action, dtype=self.dtype, device=self.device).reshape(1, -1)))
                    t = torch.cat((t, torch.tensor([R.shape[0]], device=self.device)))
                    sum_r += r
                    length += 1

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
    
    def bm_online_evaluate(self, num_eval_episodes = 100):
        
        # copied from experiment
        def eval_episodes(target_rew):
            def fn(model):
                returns, lengths = [], []
                for _ in range(num_eval_episodes):
                    with torch.no_grad():
                        
                        ret, length = evaluate_episode_rtg(
                            self.env,
                            self.input_dim,
                            self.output_dim,
                            self.model,
                            max_ep_len=self.max_ep_len,
                            scale=self.scale,
                            target_return=target_rew/self.scale,
#                             mode=mode,
                            state_mean=self.state_mean,
                            state_std=self.state_std,
                            device=self.device
                        )
                        
                    returns.append(ret)
                    lengths.append(length)
                return {
                    f'target_{target_rew}_return_mean': np.mean(returns),
                    f'target_{target_rew}_return_std': np.std(returns),
                    f'target_{target_rew}_length_mean': np.mean(lengths),
                    f'target_{target_rew}_length_std': np.std(lengths),
                }
            return fn
        
        eval_fn = eval_episodes(self.target_return)
        eval_fn(self.model)

        pass

# copied from evaluation
def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cpu',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward/scale

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length
