from evaluation.RLevaluation import Evaluater
import torch
import numpy as np

class DT_Evaluater(Evaluater):
    def _evaluate(self, env, model, dtype=torch.float, itype=torch.int, normalized=False, sequence_length = 20, n_eps = 100, target_return=3600, max_ep_len=1000, scale=1000, device='cpu', state_mean=None, state_std=None, mode='normal'):
        
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.shape[0]
        if state_mean is None:
            state_mean = np.zeros(state_dim)
        if state_std is None:
            state_std = np.array([1.0])

        

        r_per_eps = np.zeros(n_eps)
        l_per_eps = np.zeros(n_eps)

        for i in range(n_eps):
#                 print('eps', i)
            # we want to input tensors here
            R = torch.tensor([[target_return/scale]], dtype=dtype, device=device)
            s = torch.tensor([env.reset()], dtype=dtype, device=device)
            a = torch.zeros((0, action_dim), dtype=dtype, device=device)
            t = torch.tensor([1], dtype=itype, device=device)
            sum_r = 0
            done = False
            length = 0
            while not done:
                # sample next action. pad action with zeros for next action?
#                 print('R', R)
#                 print('A', torch.cat((a, torch.zeros((1, action_dim), device=device))))
                action = model(R, s, torch.cat((a, torch.zeros((1, action_dim), device=device))), t)[-1]
                # print('t sh', t.shape)
                # print('R sh', R.shape)
                s_prime, r, done, _ = env.step(action.detach().numpy())

                # append new tokens
                R = torch.cat((R, torch.tensor([R[-1] - r/scale], device=device).reshape(1, -1)))
                s = torch.cat((s, torch.tensor((s_prime- state_mean) / state_std, dtype=dtype, device=device).reshape(1, -1)))
                a = torch.cat((a, torch.tensor(action, dtype=dtype, device=device).reshape(1, -1)))
                t = torch.cat((t, torch.tensor([R.shape[0]], device=device)))
                sum_r += r
                length += 1

                # print(s.shape)

                # slice out extra tokens
                R, s, a, t = R[-sequence_length:], s[-sequence_length:], a[-sequence_length + 1:], t[
                                                                                                                    -sequence_length:]
            if normalized:
                r_per_eps[i] = env.get_normalized_score(sum_r)
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
    

class BM_Evaluater(Evaluater):
    '''
    Evaluates the model based on the author's code
    '''
    def _evaluate(self, env, model, sequence_length = 20, n_eps = 100, target_return=3600, max_ep_len=1000, scale=1000, device='cpu', state_mean=None, state_std=None, mode='normal'):
        state_dim = np.prod(env.observation_space.shape)
        action_dim = env.action_space.shape[0]

        if state_mean is None:
            state_mean = np.zeros(state_dim)
        if state_std is None:
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
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            sequence_length=sequence_length
                        )
                        
                    returns.append(ret)
                    lengths.append(length)
                return np.array(returns), np.array(lengths)
                # {
                #     # f'target_{target_rew}_return_mean': np.mean(returns),
                #     # f'target_{target_rew}_return_std': np.std(returns),
                #     # f'target_{target_rew}_length_mean': np.mean(lengths),
                #     # f'target_{target_rew}_length_std': np.std(lengths),
                    
                # }
            return fn
        
        eval_fn = eval_episodes(target_return)
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