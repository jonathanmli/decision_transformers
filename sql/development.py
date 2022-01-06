# %%
from DTagents import *
import random
import torch

# %%
# set seed for reproducbility
SEED = 6
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# %%
# make the experiment environment and dt agent
# new_batch, env, max_ep_len, scale, env_target, state_mean, state_std = prepare_experiment('gym-experiment', device='cuda')
dta = DecisionTransformerAgent(lr=0.001, warmup_steps=100, env_name='hopper')

# %%
print("Train")
dta.train(100)

# %%
print("BM Evaluate")
dta.evaluate(100)

# %%
print("Evaluate")
dta.set_evaluater()
dta.evaluate(100)

# print(dta.evaluater.all_rewards)