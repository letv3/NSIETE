import random
from collections import deque
from random import choices
import numpy as np



class Memory:
    transition = np.dtype(
            [('s', np.float64, (8,)), ('a', np.float64, (2,)),
             ('r', np.float64, (1,)), ('s_n', np.float64, (8,)), ("d", np.float64, (1,))])

    def __init__(self, buffer_capacity):
        # self.counter = 0
        # self.buffer_capacity = buffer_capacity
        # self.buffer = np.empty(buffer_capacity, dtype=self.transition)
        self.queue = deque(maxlen=buffer_capacity)

    def store(self, transition):
        self.queue.append(transition)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = choices(self.queue, k=batch_size)

        for experience in batch:
            state_batch.append(experience[0])
            action_batch.append(experience[1])
            reward_batch.append(experience[2])
            next_state_batch.append(experience[3])
            done_batch.append(experience[4])

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.queue)

    def size (self):
        return len(self.queue)
