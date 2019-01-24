import unittest
import numpy as np
import torch
from numpy.testing import assert_array_equal

from banananav.qmodel import BananaQModel

STATE_SIZE, ACTION_SIZE, BATCH_SIZE = 8, 4, 10

class TestBananaQModelModel(unittest.TestCase):

    def setUp(self):
        self.model = BananaQModel(STATE_SIZE, ACTION_SIZE)

    def test_shapes(self):
        x = self.model(torch.rand(BATCH_SIZE, STATE_SIZE))

        self.assertEqual(x.size(), (BATCH_SIZE, ACTION_SIZE))
