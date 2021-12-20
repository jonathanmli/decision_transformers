# %%
from jonathans_experiment import prepare_experiment
from DTagents import *

# %%
# make the experiment environment and dt agent
# new_batch, env, max_ep_len, scale, env_target, state_mean, state_std = prepare_experiment('gym-experiment', device='cuda')
dta = DecisionTransformerAgent()


# %%
print("Train")
dta.train(100)

# %%
print("Evaluate")
dta.evaluate(100)