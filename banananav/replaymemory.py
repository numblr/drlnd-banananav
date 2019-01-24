import random

from banananav.environment import BananaEnv


class ReplayMemory():
    def __init__(self, size, seed):
        # TODO seed
        self._buffer = [None] * size
        self._pos = 0
        self._filled = False

    def add(self, state, action, reward, next_state, is_terminal):
        self._buffer[self._pos] = (state, action, reward, next_state, is_terminal)
        self._pos = (self._pos + 1) % self.capacity()

        if self._pos == 0:
            self._filled = True

    def sample(self, size=64):
        load = self.size()
        if size > load:
            raise ValueError("size was less then the size of the memory: " + str(size))

        return random.sample(self._buffer[:load], size)

    def size(self):
        return self.capacity() if self._filled else self._pos

    def capacity(self):
        return len(self._buffer)
