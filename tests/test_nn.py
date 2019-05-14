import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../datasets")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../utils")

import  nn
from nn import architecture as nna
import torch as pt
import unittest
from loader import *
from data_divider import *
from plot_cost import *
import matplotlib.pyplot as plt

import mnist.input_data as data_source


def mnist_data():
    mnist = data_source.read_data_sets("datasets/mnist/data/", one_hot=True)
    device = nn.get_device()

    X_train = nn.from_numpy(mnist.train.images, device)#[:, 0:3000]
    Y_train = nn.from_numpy(mnist.train.labels, device)#[:, 0:3000]

    X_test = nn.from_numpy(mnist.test.images, device)
    Y_test = nn.from_numpy(mnist.test.labels, device)

    X_validation = nn.from_numpy(mnist.validation.images, device)
    Y_validation = nn.from_numpy(mnist.validation.labels, device)

    print("\ntrain shape: X=%s, Y=%s" % (X_train.shape, Y_train.shape))
    print("validation shape: X=%s, Y=%s" % (X_validation.shape, Y_validation.shape))
    print("test shape: X=%s, Y=%s" % (X_test.shape, Y_test.shape))

    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation

print("\n------> Test softmax nn: MNIST")
        
X_train, Y_train, X_test, Y_test, X_validation, Y_validation = mnist_data()

X_train = X_train#[:, 0:4000]
Y_train = Y_train#[:, 0:4000]

n = X_train.shape[1]
relu = nna.relu()

X = nna.input(n)
X = nna.layer(50)(X)
X = nna.batch_norm()(X)
X = relu(X)
X = nna.layer(20)(X)
X = relu(X)
X = nna.layer(10)(X)
X = nna.softmax()(X)

loss = nn.loss.categorical_crossentropy

gd = nn.gradient_descent(loss, 5000)
sgd = nn.stochastic(loss, 100)
momentum = nn.momentum(loss, 10)
rms = nn.rms_prop(loss, 10)
adam = nn.adam(loss, 7)

model = nn.Model(X, adam)

print("fitting model....")

parameters, cost_evaluator = model.fit(X_train, Y_train, True)

train_acc = model.evaluate(X_train, Y_train, 'train')
test_acc = model.evaluate(X_test, Y_test, 'test')

# cost_evaluator.plot_cost()
