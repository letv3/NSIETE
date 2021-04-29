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
    agent = Agent(env.env)

    reward_history = []
    moving_reward_history = []
    max_reward = 0
    average_reward = 0
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
        moving_reward = 0
        state = np.float64(env.reset())
        # state = torch.from_numpy(state).to(device)

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
            moving_reward += reward
            state = new_state
            if done:
                break

        average_reward = average_reward * 0.99 + reward * 0.01
        reward_history.append(reward)
        moving_reward_history.append(moving_reward)
        if reward > max_reward:
            max_reward = reward
        else:
            episodes_with_lower_score += 1
        #    if episodes_with_lower_score == EARLY_STOP_EPISODES:
        #        print(f"latest average revard: {score}, stopped on {episode} episode")
        #        break

        if episode % LOG_INTERVAL == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(
                 episode, reward, average_reward))

        wandb.log({
            'Episode': episode,
            'Step': t,
            'Score': reward,
            'Average Score': average_reward

        })
        # wandb.log({
        #     'Environment': [wandb.Image(env.render(mode='rgb_array'), caption='Environment')]
        # })  # https://docs.wandb.ai/guides/track/log#images-and-overlays
        # Nemusime asi tak to manualne robit, ked ze wandb priamo podporuje GYM toolkit
        # monitor_gym=True

    torch.cuda.empty_cache()
    run.finish()

            print('Ep {}\tLast score: {:.2f}\t Average score: {:.2f}\t Moving average score: {:.2f}'.format(
                 episode, reward, average_reward,moving_reward))
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
