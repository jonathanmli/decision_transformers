from evaluation.RLevaluation import Evaluater
import torch
import numpy as np

class sql_Evaluater(Evaluater):
    def __init__(self, 
            dtype=torch.float, 
            itype=torch.int, 
            normalized=True, 
            sequence_length = 20, 
            target_return=3600, 
            max_ep_len=1000, 
            scale=1000, 
            device='cpu', 
            state_mean=None, 
            state_std=None, 
            mode='normal'):

        super().__init__()
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device
        self.mode=mode
        self.sequence_length=sequence_length
        self.target_return=target_return
        self.max_ep_len=max_ep_len
        self.scale=scale
        self.dtype=dtype
        self.itype=itype
        self.normalized=normalized

class DT_Evaluater(sql_Evaluater):

    def _evaluate(self, env, model, n_eps = 100):
        
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.shape[0]
        if self.state_mean is None:
            state_mean = np.zeros(state_dim)
        if self.state_std is None:
            state_std = np.array([1.0])

        

        r_per_eps = np.zeros(n_eps)
        l_per_eps = np.zeros(n_eps)

        for i in range(n_eps):
#                 print('eps', i)
            # we want to input tensors here
            R = torch.tensor([[self.target_return/self.scale]], dtype=self.dtype, device=self.device)
            s = torch.tensor([env.reset()], dtype=self.dtype, device=self.device)
            a = torch.zeros((0, action_dim), dtype=self.dtype, device=self.device)
            t = torch.tensor([1], dtype=self.itype, device=self.device)
            sum_r = 0
            done = False
            length = 0
            while not done:
                # sample next action. pad action with zeros for next action?
#                 print('R', R)
#                 print('A', torch.cat((a, torch.zeros((1, action_dim), device=device))))
                action = model.get_action(s, torch.cat((a, torch.zeros((1, action_dim), device=self.device))), None, R, t)
                # model(R, s, torch.cat((a, torch.zeros((1, action_dim), device=self.device))), t)[-1]
                # print('t sh', t.shape)
                # print('R sh', R.shape)
                s_prime, r, done, _ = env.step(action.cpu().detach().numpy())

                # append new tokens
                R = torch.cat((R, torch.tensor([R[-1] - r/self.scale], device=self.device).reshape(1, -1)))
                s = torch.cat((s, torch.tensor((s_prime- self.state_mean) / self.state_std, dtype=self.dtype, device=self.device).reshape(1, -1)))
                a = torch.cat((a, torch.tensor(action, dtype=self.dtype, device=self.device).reshape(1, -1)))
                t = torch.cat((t, torch.tensor([R.shape[0]], device=self.device)))
                sum_r += r
                length += 1

                # print(s.shape)

                # slice out extra tokens
                R, s, a, t = R[-self.sequence_length:], s[-self.sequence_length:], a[-self.sequence_length + 1:], t[-self.sequence_length:]
            if self.normalized:
                r_per_eps[i] = 100*env.get_normalized_score(sum_r)
            else:
                r_per_eps[i] = sum_r

            l_per_eps[i] = length
        #         , R, s, a, t
        # print('mean return', r_per_eps.mean())
        # print('std returns', r_per_eps.std())
        # print('mean lengths', l_per_eps.mean())
        # print('std lengths', l_per_eps.std())
        self.record(r_per_eps, l_per_eps)
        return r_per_eps, l_per_eps
    

class BM_Evaluater(sql_Evaluater):
    '''
    Evaluates the model based on the author's code
    '''
    def _evaluate(self, env, model, n_eps=100):
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.shape[0]

        if self.state_mean is None:
            state_mean = np.zeros(state_dim)
        if self.state_std is None:
            state_std = np.array([1.0])
        # copied from dt experiment
        def eval_episodes(target_rew):
            def fn(model):
                returns, lengths = [], []
                for _ in range(n_eps):
                    with torch.no_grad():
                        # print(max_ep_len)
                        ret, length = self.evaluate_episode_rtg(
                            env,
                            state_dim,
                            action_dim,
                            model,
                            max_ep_len=self.max_ep_len,
                            scale=self.scale,
                            target_return=target_rew/self.scale,
                            mode=self.mode,
                            state_mean=self.state_mean,
                            state_std=self.state_std,
                            device=self.device,
                            sequence_length=self.sequence_length
                        )
                        
                    returns.append(ret)
                    lengths.append(length)

                # original code prints mean and std here. our code has a wrapper that does this instead
                print({
                    f'target_{target_rew}_return_mean': np.mean(returns),
                    f'target_{target_rew}_return_std': np.std(returns),
                    f'target_{target_rew}_length_mean': np.mean(lengths),
                    f'target_{target_rew}_length_std': np.std(lengths),
                    
                })
                return np.array(returns), np.array(lengths)

                
            return fn
        
        eval_fn = eval_episodes(self.target_return)
        ret, len = eval_fn(model)
        self.record(ret, len)
        return ret, len


    # copied from dt evaluation
    def evaluate_episode_rtg(
            self,
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
            sequence_length=20
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

            # print(timesteps[-sequence_length:].shape)
            # print(states[-sequence_length:].shape)

            # I modified this part to slice to the K that we trained on. 
            # I believe the authors did something similar but in models.get_action instead
            # Also converted tensors to correct dimensions
            action = model.get_action(
                ((states.to(dtype=torch.float32) - state_mean) / state_std)[-sequence_length:],
                actions.to(dtype=torch.float32)[-sequence_length:],
                rewards.to(dtype=torch.float32)[-sequence_length:],
                target_return.to(dtype=torch.float32).reshape(-1,1)[-sequence_length:],
                timesteps.to(dtype=torch.long).reshape(-1)[-sequence_length:],
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