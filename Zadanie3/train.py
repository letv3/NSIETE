import numpy as np
import torch
from src.Env import Env
from src.Model import Agent
import wandb
NUM_EPISODES = 8000  # Number of episodes
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

    learning_rate = 1e-4
    gamma = 0.99
    tau = 1e-2

    # Wandb init
    # run = wandb.init(project='', entity='xpetricko', monitor_gym=True)
    run = wandb.init(project='zadani3-lunarlander', entity='lytyvnol', monitor_gym=True)
    run.name = f"Test-run-{NUM_EPISODES}"

    env = Env()
    agent = Agent(env.env, learning_rate, gamma, tau)

    reward_history = []
    max_score = 0
    average_score = 0
    episodes_with_lower_score = 0

    # Wandb config
    config = wandb.config
    config.learing_rate = learning_rate
    config.gamma = gamma
    config.tau = tau
    config.clip_parameter = agent.clip_param
    config.buffer_capacity = agent.buffer_capacity
    config.batch_size = agent.batch_size

    for episode in range(NUM_EPISODES):
        score = 0
        state = np.float64(env.reset())

        # Actions in one episode
        for t in range(MAX_STEPS):
            action = agent.select_action(state)
            new_state, reward, done = env.step(action)
            new_state = np.float64(new_state)
            if RENDER:
                env.render()
            if agent.memory.store((state, action, reward, new_state, done)):
                print('Updating model.')
                agent.update()
            score += reward
            state = new_state
            if done:
                break

        average_score = average_score * 0.99 + score * 0.01
        reward_history.append(score)
        if score > max_score:
            max_score = score
        else:
            episodes_with_lower_score += 1
        #    if episodes_with_lower_score == EARLY_STOP_EPISODES:
        #        print(f"latest average revard: {score}, stopped on {episode} episode")
        #        break

        if episode % LOG_INTERVAL == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(
                 episode, score, average_score))

        wandb.log({
            'Episode': episode,
            'Step': t,
            'Score': score,
            'Average Score': average_score

        })
        # wandb.log({
        #     'Environment': [wandb.Image(env.render(mode='rgb_array'), caption='Environment')]
        # })  # https://docs.wandb.ai/guides/track/log#images-and-overlays
        # Nemusime asi tak to manualne robit, ked ze wandb priamo podporuje GYM toolkit
        # monitor_gym=True

    torch.cuda.empty_cache()
    run.finish()

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
