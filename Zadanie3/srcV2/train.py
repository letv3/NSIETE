import numpy as np
import torch as T
from Agent import Agent
from Noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
import os
import argparse
import json
import gym
import random
from collections import deque

from srcV2.Env import LunarLanderContinuous as LLC
from srcV2.Misc import *

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

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

    env = LLC()

    T.manual_seed(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    # print(f"================= {'Environment Information'.center(30)} =================")
    # print(f"Action space shape: {env.action_space.shape}")
    # print(f"Action space upper bound: {env.action_space.high}")
    # print(f"Action space lower bound: {env.action_space.low}")
    #
    # print(f"stateervation space shape: {env.observation_space.shape}")
    # print(f"stateervation space upper bound: {np.max(env.observation_space.high)}")
    # print(f"stateervation space lower bound: {np.min(env.observation_space.low)}")
    #
    # print(f"================= {'Parameters'.center(30)} =================")
    # for k, v in __dict__.items():
    #     print(f"{k:<20}: {v}")

    # Experiment directory storage
    # counter = 1
    # env_path = os.path.join("experiments", env)
    # if not os.path.exists(env_path):
    #     os.mkdir(env_path)
    #
    # while True:
    #     try:
    #         experiment_path = os.path.join(env_path, f"{experiment_name}_{counter}")
    #         os.mkdir(experiment_path)
    #         os.mkdir(os.path.join(experiment_path, "saves"))
    #         break
    #     except FileExistsError as e:
    #         counter += 1
    #
    # with open(os.path.join(experiment_path, 'parameters.json'), 'w') as f:
    #     json.dump(__dict__, f, indent=2)

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

    counter = 0
    reward_history = deque(maxlen=100)
    loss_history = deque(maxlen=50)

    for episode in range(episodes):
        state = env.reset()
        # noise.reset()
        episode_reward = 0.0
        actor_loss = 0.0
        critic_loss = 0.0

        # Generate rollout and train agent
        for step in range(max_steps):

            if render:
                env.render()

            # Get actions
            with T.no_grad():
                if not exploration_prob:
                    if episode >= exploration:
                        action = agent.action(state)  # TODO + T.tensor(noise(), dtype=T.float, device=device)
                        action = T.clamp(action, -1.0, 1.0)
                    else:
                        action = agent.random_action()
                else:
                    if np.random.random() < exploration_prob:
                        action = agent.action(state)  # TODO + T.tensor(noise(), dtype=T.float, device=device)
                        action = T.clamp(action, -1.0, 1.0)
                    else:
                        action = agent.random_action()
                        # TODO exploration_prob calculation

            # Take step in environment
            new_state, reward, done, _ = env.step(action.detach().cpu().numpy() * env.action_space.high)
            episode_reward += reward

            # Store experience
            agent.store_experience((state, action.detach().cpu().numpy(), reward, new_state, done))

            # Train agent
            if counter % train_interval == 0:
                if agent.memory.size() > agent.min_replay_size:
                    counter = 0
                    loss = agent.train()
                    # Change probability of random action
                    exploration_prob = calculate_exploration_prob(loss_history, exploration_prob, 10)
                    # Loss information kept for monitoring purposes during training
                    actor_loss += loss['actor_loss']
                    critic_loss += loss['critic_loss']
                    loss_history.append(critic_loss)

                    agent.update()

            state = new_state
            counter += 1
            if done:
                break

        reward_history.append(episode_reward)
        print(f"Episode: {episode} Episode reward: {episode_reward} Average reward: {np.mean(reward_history)}")
        # print(f"Actor loss: {actor_loss/(step/train_interval)} Critic loss: {critic_loss/(step/train_interval)}")

        # Evaluate
        if episode % eval_interval == 0:
            evaluation_rewards = 0
            for evalutaion_episode in range(eval_eps):
                state = env.reset()
                rewards = 0

                for step in range(max_steps):
                    if render:
                        env.render()

                    # Get actions
                    with T.no_grad():
                        action = agent.action(state)

                    # Take step in environment
                    new_state, reward, done, _ = env.step(action.detach().cpu().numpy() * env.action_space.high)

                    # Update state
                    state = new_state

                    # Update rewards
                    rewards += reward

                    # End episode if done
                    if done:
                        break

                evaluation_rewards += rewards

            evaluation_rewards = round(evaluation_rewards / eval_eps, 3)
            # save_path = os.path.join(experiment_path, "saves")
            # agent.save_agent(save_path)
            print(f"Episode: {episode} Average evaluation reward: {evaluation_rewards}")
            # with open(f"{experiment_path}/evaluation_rewards.csv", "a") as f:
            #     f.write(f"{episode}, {evaluation_rewards}\n")
            # try:
            #     if evaluation_rewards > env.spec.reward_threshold * 1.1: # x 1.1 because of small eval_episodes
            #         print(f"Environment solved after {episode} episodes")
            #         break
            # except Exception as e:
            #     if evaluation_rewards > -120:
            #         print(f"Environment solved after {episode} episodes")
            #         break
