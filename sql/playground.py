import gym
import d4rl # Import required to register environments

# Create the environment
env = gym.make('hopper-medium-v2')

# d4rl abides by the OpenAI gym interface
env.reset()
R = 0.0
for i in range(100):
    s, r, d, _ = env.step(env.action_space.sample())
    R += r

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations
print(R)

# # Alternatively, use d4rl.qlearning_dataset which
# # also adds next_observations.
# dataset = d4rl.qlearning_dataset(env)

print(env.get_normalized_score(1700))
