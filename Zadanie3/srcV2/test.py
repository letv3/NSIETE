import numpy as np
import torch
from Agent import Agent
import argparse
import json
import gym
import random
import torch as T

from srcV2.Env import LunarLanderContinuous as LLC
from srcV2.Misc import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

experiment_name = "Zadanie 3"
episodes = 10000
max_steps = 1000
exploration = 0
exploration_prob = 0
train_interval = 1
eval_eps = 5
eval_interval = 10

batch_size = 64
min_replay_size = 1000
memory_capacity = 1000000
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.99
tau = 0.005
render = False
gaussian_noise = False
noise_param = 0
seed = 0
#Exploration mode
# 0- no exploration, 1- static episode eploration, 2 - variable critic loss exploration
exploration_mode = 1


if __name__ == "__main__":
   
    # Load json parameters
   

    env = LLC()

    T.manual_seed(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

   

    n_actions = env.action_space.shape[0] if type(env.action_space) == gym.spaces.box.Box else env.action_space.n

    agent = Agent(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        memory_capacity=memory_capacity,
        tau=tau,
        n_inputs=env.observation_space.shape,
        n_actions=n_actions
    )

    agent.load_agent(experiment_name)

    total_rewards = 0.0

    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0.0

        for step in range(max_steps):
            if render:
                env.render()

            # Get actions
            with torch.no_grad():
                action = agent.action(obs)

            # Take step in environment
            new_obs, reward, done, _ = env.step(action.detach().cpu().numpy() * env.action_space.high)

            # Update obs
            obs = new_obs
            # Update rewards
            episode_reward += reward
            # End episode if done
            if done:
                break

        total_rewards += episode_reward
        episode_reward = round(episode_reward, 3)
        print(f"Episode: {episode} Evaluation reward: {episode_reward}")

    print(f"{episodes} episode average: {round(total_rewards / episodes, 3)}")