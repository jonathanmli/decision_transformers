# %%
from DTagents import *

# %%
# make the experiment environment and dt agent
# new_batch, env, max_ep_len, scale, env_target, state_mean, state_std = prepare_experiment('gym-experiment', device='cuda')
dta = DecisionTransformerAgent(lr=0.001, warmup_steps=1000)

# %%
print("Train")
dta.train(1000)

# %%
print("BM Evaluate")
dta.evaluate(100)

# %%
print("Evaluate")
dta.set_evaluater()
dta.evaluate(100)