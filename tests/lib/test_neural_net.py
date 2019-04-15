
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../lib")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../datasets")

import neural_net as nn
import torch as pt
import unittest

import mnist.input_data as data_source

class TestNeuralNet(unittest.TestCase):

    def test_Input(self):
        X = nn.Input(100)

        self.assertEqual(X["layers"][0]["size"], 100)
    
    def _init_params(self):
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

    def test_neural_net(self):
        # source reference: https://github.com/llSourcell/tensorflow_demo/blob/master/board.py

        mnist = data_source.read_data_sets("datasets/mnist/data/", one_hot=True)
        print("train shape: ", mnist.train.images.shape)
        print("validation shape: ",  mnist.validation.images.shape)
        print("test shape", mnist.test.images.shape)

        # print(mnist.train.images[0])
        # print("\n", mnist.train.labels[0])

        learning_rate = 0.01
        training_iteration = 30
        batch_size = 100
        display_step = 2
        n = 784

        X = nn.Input(n)
        X = nn.Layer(50, "relu")(X)
        X = nn.Layer(20, "relu")(X)
        X = nn.Layer(10, "softmax")(X)



    

if __name__ == "__main__":
    unittest.main()

