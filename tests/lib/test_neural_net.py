
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../lib")

import neural_net as nn
import torch as pt
import unittest

class TestNeuralNet(unittest.TestCase):

    def test_Input(self):
        X = nn.Input(100)

        self.assertEqual(X["layers"][0]["size"], 100)
    
    def test_init_params(self):
        X = nn.Input(20)
        X = nn.Layer(50, "relu")(X)
        X = nn.Layer(22, "relu")(X)
        X = nn.Layer(1, "sigmoid")(X)

        parameters = nn.init_params(X["layers"])

        self.assertEqual(len(parameters), 6)
        self.assertEqual(parameters["W1"].shape, (50, 20))
        self.assertEqual(parameters["W2"].shape, (22, 50))
        self.assertEqual(parameters["W3"].shape, (1, 22))
        self.assertEqual(parameters["b1"].shape, (50, 1))
        self.assertEqual(parameters["b2"].shape, (22, 1))
        self.assertEqual(parameters["b3"].shape, (1, 1))

if __name__ == "__main__":
    unittest.main()

