import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../lib")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../datasets")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../utils")

import  nn
from nn import architecture as nna
import torch as pt
import unittest
from loader import *
from data_divider import *
from plot_cost import *
import matplotlib.pyplot as plt

from nn.parameters import initialize_parameters


class TestNeuralNet(unittest.TestCase):

    def test_parameters(self):
        X = nna.input(100)
        X = nna.layer(50)(X)
        X = nna.relu()(X)
        X = nna.layer(20)(X)
        X = nna.relu()(X)
        X = nna.layer(10)(X)
        X = nna.softmax()(X)

        layers = X['layers']

        params = initialize_parameters(layers, nn.get_device())

        p1, p_next = params
        W1, b1 = p1
        p2, p_next = p_next
        W2, b2 = p2
        p3, p_next = p_next
        W3, b3 = p3

        self.assertEqual(W1.shape, (50, 100))
        self.assertEqual(b1.shape, (50, 1))
        self.assertEqual(W2.shape, (20, 50))
        self.assertEqual(b2.shape, (20, 1))
        self.assertEqual(W3.shape, (10, 20))
        self.assertEqual(b3.shape, (10, 1))
        self.assertEqual(p_next, None)


if __name__ == "__main__":
    unittest.main()


