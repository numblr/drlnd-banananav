import unittest
import numpy as np
import torch

from banananav.environment import BananaEnv, BananaAgent

# class TestBananaEnv(unittest.TestCase):
#     def setUp(self):
#         self.env = BananaEnv()
#
#     def test_action_size(self):
#         self.assertEqual(self.env.get_action_size(), 4)
#
#     def test_state_size(self):
#         self.assertEqual(self.env.get_action_size(), 37)

class TestBananaAgent(unittest.TestCase):
    def test_act_draws_from_Q(self):
        dummy_Q = lambda s: (np.arange(4) == s[0]).astype(float)
        self.agent = BananaAgent(dummy_Q, 4)

        self.assertEqual(self.agent.act([0]), 0)
        self.assertEqual(self.agent.act([1]), 1)
        self.assertEqual(self.agent.act([2]), 2)
        self.assertEqual(self.agent.act([3]), 3)

    def test_act_draws_random(self):
        dummy_Q = lambda s: (np.arange(100) == s[0]).astype(float)
        self.agent = BananaAgent(dummy_Q, 100, 1.0)

        test = [ self.agent.act([0]) for _ in range(100) ]
        self.assertFalse(len(list(set(test))) == 1)

    def test_numpy_unsqueezed(self):
        rand_Q = lambda s: np.random.rand(1,4)

        self.agent = BananaAgent(rand_Q, 4)

        self.assertIsInstance(self.agent.act([0]), int)

    def test_numpy_squeezed(self):
        rand_Q = lambda s: np.random.rand(4)

        self.agent = BananaAgent(rand_Q, 4)

        self.assertIsInstance(self.agent.act([0]), int)

    def test_unsqueezed_tensor(self):
        rand_Q = lambda s: torch.rand((1,4))

        self.agent = BananaAgent(rand_Q, 4)

        self.assertIsInstance(self.agent.act([0]), int)

    def test_squeezed_tensor(self):
        rand_Q = lambda s: torch.rand((4,))

        self.agent = BananaAgent(rand_Q, 4)

        self.assertIsInstance(self.agent.act([0]), int)
