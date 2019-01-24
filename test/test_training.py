import unittest
import numpy as np
import torch
from numpy.testing import assert_array_equal

from banananav.training import DeepQLearner
from banananav.replaymemory import ReplayMemory

STATE_SIZE, ACTION_SIZE, BATCH_STEPS, BATCH_SIZE = 3, 4, 2, 2

class TestDeepQLearner(unittest.TestCase):

    def setUp(self):
        self.learner = DeepQLearner(batch_steps=BATCH_STEPS, batch_size=BATCH_SIZE,
                env=DummyEnv(), memory=ReplayMemory(2))

    def test_learner_executes_training(self):
        training_info = [ info for info in self.learner.train(3) ]

        self.assertEqual(len(training_info), 8)
        self.assertEqual(len([i for i in training_info if i[2]]), 3)


class DummyEnv():
    def get_state_size(self):
        return STATE_SIZE

    def get_action_size(self):
        return ACTION_SIZE

    def generate_episode(self, actor, train_mode=False):
        return ( state_data for state_data in [
            (np.array([1.0,1.0,1.0]), 0, -1.0, np.array([1.0,1.0,1.0]), False),
            (np.array([1.0,1.0,1.0]), 1, 1.0, np.array([1.0,1.0,1.0]), False),
            (np.array([1.0,1.0,1.0]), 2, 2.0, np.array([1.0,1.0,1.0]), False),
            (np.array([1.0,1.0,1.0]), 1, 1.0, np.array([1.0,1.0,1.0]), False),
            (np.array([1.0,1.0,1.0]), 0, 1.0, np.array([1.0,1.0,1.0]), True)
        ])

    def get_score(self):
        return 1
