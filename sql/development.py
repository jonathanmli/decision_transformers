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
dtc = DTConfiguration(lr=0.001, warmup_steps=100, training_steps=100, env_name='hopper')
dta = DecisionTransformerAgent(dtc)

# %%
print("Train")
dta.train(100)

# %%
print("BM Evaluate")
dta.evaluate(10)

# %%
print("Evaluate")
dta.set_evaluater()
dta.evaluate(10)

# print(dta.evaluater.all_rewards)