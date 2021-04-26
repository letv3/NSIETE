import random
from collections import deque
import numpy as np


class Memory:
    def __init__(self, buffer_capacity, transition):
        self.counter = 0
        self.buffer_capacity = buffer_capacity
        self.buffer = np.empty(buffer_capacity, dtype=transition)

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = np.random.choice(self.buffer, batch_size)

        for experience in batch:
            state_batch.append(experience["s"])
            action_batch.append(experience["a"])
            reward_batch.append(experience["r"])
            next_state_batch.append(experience["s_n"])
            done_batch.append(experience["d"])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return self.counter
