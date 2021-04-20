import numpy as np

import torch


from src.env import Env, EnvD
from src.model import Agent, AgentD


import matplotlib.pyplot as plt



NUM_EPISODES = 100000
MAX_STEPS = 2000


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

if __name__ == "__main__":

    agent = Agent()
    env = Env()

    state = env.reset()

    reward_history = []
    average_reward = 0

    for episode in range(NUM_EPISODES):
        score = 0
        state = env.reset()

        # Cycle for one opisode
        for t in range(MAX_STEPS):
            action, a_logp = agent.select_action(state)
            new_state, reward, done, die = env.step(action)
            if False:
                env.render()
            if agent.store((state, action, a_logp, reward, new_state)):
                print('Updating model.')
                agent.update()
            score += reward
            state = new_state
            if done or die:
                break

        average_reward = average_reward * 0.99 + score * 0.01
        reward_history.append(score)


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