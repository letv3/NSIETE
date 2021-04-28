from src.LunarLander import LunarLander
import numpy as np


class Env:
    def __init__(self, continuous=True):
        self.continuous = continuous
        self.seed = 0
        self.env = LunarLander(continuous=continuous)
        self.env.seed(self.seed)

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        # self.die = False
        observation = self.env.reset()
        return observation

    def step(self, action):
        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    def render(self, *args, **kvargs):
        return self.env.render(*args, **kvargs)

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


if __name__ == '__main__':
    pass
