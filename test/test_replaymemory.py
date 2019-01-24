import unittest
import numpy as np
from numpy.testing import assert_array_equal

from banananav.replaymemory import ReplayMemory

class TestReplayMemory(unittest.TestCase):
    def setUp(self):
        self.memory = ReplayMemory(3, 0)

    def test_add(self):
        self.memory.add((11,21), 1, 1, (31,41), False)

        self.assertEqual(self.memory.capacity(), 3)
        self.assertEqual(self.memory.size(), 1)
        assert_array_equal(self.memory.sample(1),
            [((11,21), 1, 1, (31,41), False)])

    def test_add_less_than_capacity(self):
        self.memory.add((11,21), 1, 1, (31,41), False)
        self.memory.add((12,22), 2, 2, (32,42), False)

        self.assertEquals(self.memory.size(), 2)

        _, actions, _, _, _ = zip(*self.memory.sample(2))
        self.assertEqual(len(actions), 2)
        self.assertEqual(max(actions), 2)
        self.assertEqual(min(actions), 1)

    def test_add_capacity(self):
        self.memory.add((11,21), 1, 1, (31,41), False)
        self.memory.add((12,22), 2, 2, (32,42), False)
        self.memory.add((12,23), 3, 3, (33,43), False)

        self.assertEquals(self.memory.size(), 3)

        _, actions, _, _, _ = zip(*self.memory.sample(3))
        self.assertEqual(len(actions), 3)
        self.assertEqual(max(actions), 3)
        self.assertEqual(min(actions), 1)
        self.assertTrue(2 in actions)

    def test_add_more_than_capacity(self):
        self.memory.add((11,21), 1, 1, (31,41), False)
        self.memory.add((12,22), 2, 2, (32,42), False)
        self.memory.add((13,23), 3, 3, (33,43), False)
        self.memory.add((14,24), 4, 4, (34,44), False)

        self.assertEquals(self.memory.size(), 3)

        _, actions, _, _, _ = zip(*self.memory.sample(3))
        self.assertEqual(len(actions), 3)
        self.assertEqual(max(actions), 4)
        self.assertEqual(min(actions), 2)
        self.assertTrue(3 in actions)

    def test_add_is_fifo(self):
        self.memory.add((11,21), 1, 1, (31,41), False)
        self.memory.add((12,22), 2, 2, (32,42), False)
        self.memory.add((13,23), 3, 3, (33,43), False)
        self.memory.add((14,24), 4, 4, (34,44), False)
        self.memory.add((15,25), 5, 5, (35,45), False)
        self.memory.add((16,26), 6, 6, (36,46), False)

        self.assertEquals(self.memory.size(), 3)

        _, actions, _, _, _ = zip(*self.memory.sample(3))
        self.assertEqual(len(actions), 3)
        self.assertEqual(max(actions), 6)
        self.assertEqual(min(actions), 4)
        self.assertTrue(5 in actions)

    def test_sample_is_random(self):
        self.memory.add((11,21), 1, 1, (31,41), False)
        self.memory.add((12,22), 2, 2, (32,42), False)
        self.memory.add((13,23), 3, 3, (33,43), False)

        _, actions, _, _, _ = zip(*sum([ self.memory.sample(2) for _ in range(100) ],[]))
        self.assertEqual(len(actions), 200)
        self.assertEqual(len(set(actions)), 3)

    def test_sample_is_without_replacement(self):
        self.memory.add((11,21), 1, 1, (31,41), False)
        self.memory.add((12,22), 2, 2, (32,42), False)
        self.memory.add((13,23), 3, 3, (33,43), False)

        _, actions, _, _, _ = zip(*sum([ self.memory.sample(3) for _ in range(100) ],[]))
        self.assertEqual(len(actions), 300)
        self.assertEqual(len(set(actions)), 3)
        self.assertEqual(len([a for a in actions if a == 1]), 100)
        self.assertEqual(len([a for a in actions if a == 2]), 100)
