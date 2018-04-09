import random

import numpy as np


class ExperienceBuffer(object):
    """
    Here we can store [state, action, reward, finished, state_out]
    after each episode and get samples for training
    """

    def __init__(self, max_size):
        super(ExperienceBuffer, self).__init__()
        self.buffer = []
        self.max_size = max_size

    def append(self, experience):
        self.buffer.append(experience)

        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def extend(self, experiences):
        self.buffer.extend(experiences)
        self.buffer[:-self.max_size] = []

    def sample(self, size):
        length = len(self.buffer)
        if size > length:
            size = length
        return np.array(random.sample(self.buffer, size))

    def clear(self):
        self.buffer.clear()

    def __repr__(self):
        return 'Experience Buffer with %d elements (max: %d)' % (len(self.buffer), self.max_size)

    def __len__(self):
        return len(self.buffer)
