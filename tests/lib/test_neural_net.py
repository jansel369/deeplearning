
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../lib")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../datasets")

import neural_net as nn
from neural_net import architecture as nna
import torch as pt
import unittest

import mnist.input_data as data_source

# def toTensor(np)

class TestNeuralNet(unittest.TestCase):

    # def test_Input(self):
    #     X = nna.input(100)

    #     self.assertEqual(X["layers"][0]["size"], 100)
    
    # def _init_params(self):
    #     X = nna.input(20)
    #     X = nna.layer(50, "relu")(X)
    #     X = nna.layer(22, "relu")(X)
    #     X = nna.layer(1, "sigmoid")(X)

    #     parameters = nn.init_params(X["layers"])

    #     self.assertEqual(len(parameters), 6)
    #     self.assertEqual(parameters["W1"].shape, (50, 20))
    #     self.assertEqual(parameters["W2"].shape, (22, 50))
    #     self.assertEqual(parameters["W3"].shape, (1, 22))
    #     self.assertEqual(parameters["b1"].shape, (50, 1))
    #     self.assertEqual(parameters["b2"].shape, (22, 1))
    #     self.assertEqual(parameters["b3"].shape, (1, 1))

    def test_neural_net(self):
        # source reference: https://github.com/llSourcell/tensorflow_demo/blob/master/board.py

        mnist = data_source.read_data_sets("datasets/mnist/data/", one_hot=True)
        device = nn.get_device()

        X_train = nn.from_numpy(mnist.train.images.T, device)[:, 0:2000]
        Y_train = nn.from_numpy(mnist.train.labels.T, device)[:, 0:2000]

        X_test = nn.from_numpy(mnist.test.images.T, device)
        Y_test = nn.from_numpy(mnist.test.labels.T, device)

        X_validation = nn.from_numpy(mnist.validation.images.T, device)
        Y_validation = nn.from_numpy(mnist.validation.labels.T, device)

        print("\ntrain shape: X=%s, Y=%s" % (X_train.shape, Y_train.shape))
        print("validation shape: X=%s, Y=%s" % (X_validation.shape, Y_validation.shape))
        print("test shape: X=%s, Y=%s" % (X_test.shape, Y_test.shape))

        learning_rate = 0.01
        # batch_size = 100
        # display_step = 2
        training_iteration = 800
        n = X_train.shape[0]

        X = nna.input(n)
        X = nna.dense(50, 'relu')(X)
        X = nna.dense(20, 'relu')(X)
        X = nna.dense(10, 'softmax')(X)

        # X = nn.GradientDescent(learning_rate, training_iteration)(X)
        optimizer = nn.GradientDescent(learning_rate, training_iteration, nn.categorical_crossentropy)

        # print("optimiszesr: '%s'" % optimizer.loss)

        model = nn.Model(X)

        model.optimization(optimizer)

        print("fitting model....")
        model.fit(X_train, Y_train, True, device)

        train_acc = model.evaluate(X_train, Y_train)
        test_acc = model.evaluate(X_test, Y_test)

        print("train accuracy: {} %".format(train_acc))
        print("test accuracy: {} %".format(test_acc))

if __name__ == "__main__":
    unittest.main()

