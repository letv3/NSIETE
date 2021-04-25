import numpy as np
import torch
from src.Env import Env
from src.Model import Agent
import matplotlib.pyplot as plt


NUM_EPISODES = 100000  # Number of episodes
MAX_STEPS = 2000  # Max number of steps in one episodes
EARLY_STOP_EPISODES = 10  # Stop after n episodes no progress
LOG_INTERVAL = 10  # Log after n of episodes
RENDER = True



# Torch setup
torch.manual_seed(0)
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed(0)

# Main program
if __name__ == "__main__":

    agent = Agent()
    env = Env()

    reward_history = []
    max_score = 0
    average_reward = 0
    episodes_with_lower_score = 0

    for episode in range(NUM_EPISODES):
        score = 0
        state = env.reset()

        # Actions in one episode
        for t in range(MAX_STEPS):
            action, a_logp = agent.select_action(state)
            new_state, reward, done = env.step(action)
            if RENDER:
                env.render()
            if agent.store((state, action, a_logp, reward, new_state)):
                print('Updating model.')
                agent.update()
            score += reward
            state = new_state
            if done or reward <= -200:
                break

        average_reward = average_reward * 0.99 + score * 0.01
        reward_history.append(score)
        if score > max_score:
            max_score = score
        else:
            episodes_with_lower_score += 1
            if episodes_with_lower_score == EARLY_STOP_EPISODES:
                print(f"latest average revard: {score}, stopped on {episode} episode")
                break

        # if episode % args.log_interval == 0:
        #     print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(
        #         episode, score, average_reward))
        #
        #     if args.save:
        #         agent.save_param(name=str(args.train_ord) + "_model_state")
        #         with open("data/reward/"+str(args.train_ord)+'_reward_history.txt', 'w') as f:
        #             f.seek(0)
        #             f.truncate
        #             f.writelines("%f\n" % reward for reward in reward_history)
        #
        # if args.save and average_reward > env.reward_threshold:
        #     env.reward_threshold += 10
        #     agent.save_param(name=str(args.train_ord)+"_" +
        #                      str(average_reward)+"_reward_")
        #     with open("data/reward/"+str(args.train_ord)+'_reward_history.txt', 'w') as f:
        #         f.seek(0)
        #         f.truncate
        #         f.writelines("%f\n" % reward for reward in reward_history)
