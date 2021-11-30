import numpy as np
import torch
from Agent import Agent
import argparse
import json
import gym
import random
import torch as T
import wandb

from srcV2.Env import LunarLanderContinuous as LLC
from srcV2.Misc import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

experiment_name = "Explration_100_prob_01"
episodes = 20
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
render = True
gaussian_noise = False
noise_param = 0
seed = 0
#Exploration mode
# 0- no exploration, 1- static episode eploration, 2 - variable critic loss exploration
exploration_mode = 1


wandb_log = True

run_params = {
    "experiment_name": experiment_name,
    "episodes": episodes,
    "max_steps": max_steps,
    "exploration": exploration if exploration else None,
    "exploration_prob": exploration_prob if exploration_prob else None,
    "train_interval": train_interval,
    "batch_size": batch_size,
    "min_replay_size": min_replay_size,
    "memory_capacity": memory_capacity,
    "actor_lr": actor_lr,
    "critic_lr": critic_lr,
    "gamma": gamma,
    "tau": tau,
    "gaussian_noise": gaussian_noise if gaussian_noise else None,
    "noise_param": noise_param if noise_param else None,
    "seed": seed
}


if __name__ == "__main__":

    # gym.logger.set_level(gym.logger.DEBUG)

    if wandb_log:
        # Wandb init
        run = wandb.init(project='zadani3-lunarlander', entity='lytyvnol',
                         config=run_params, monitor_gym=True)
        run.name = f"TEST2-Model:{experiment_name}"
        wandb.config.description = ""

    env = LLC()
    env = gym.wrappers.Monitor(env, "./vid",
                               force=True,
                               mode='evaluation')


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
        else:
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True

        total_rewards += episode_reward
        episode_reward = round(episode_reward, 3)
        print(f"Episode: {episode} Evaluation reward: {episode_reward}")
        if wandb_log:
            wandb.log({
                "Episode": episode,
                "Episode reward": episode_reward
            })


    print(f"{episodes} episode average: {round(total_rewards / episodes, 3)}")
    env.close()
    if wandb_log:
        run.finish()