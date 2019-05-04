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

import mnist.input_data as data_source


def mnist_data():
    mnist = data_source.read_data_sets("datasets/mnist/data/", one_hot=True)
    device = nn.get_device()

    X_train = nn.from_numpy(mnist.train.images.T, device)#[:, 0:3000]
    Y_train = nn.from_numpy(mnist.train.labels.T, device)#[:, 0:3000]

    X_test = nn.from_numpy(mnist.test.images.T, device)
    Y_test = nn.from_numpy(mnist.test.labels.T, device)

    X_validation = nn.from_numpy(mnist.validation.images.T, device)
    Y_validation = nn.from_numpy(mnist.validation.labels.T, device)

    print("\ntrain shape: X=%s, Y=%s" % (X_train.shape, Y_train.shape))
    print("validation shape: X=%s, Y=%s" % (X_validation.shape, Y_validation.shape))
    print("test shape: X=%s, Y=%s" % (X_test.shape, Y_test.shape))

    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation

def plot_cost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel("costs")
    plt.xlabel("iterations / 100s")
    plt.title("neural net (a=" + str(learning_rate) + ")")
    plt.show()

print("\n------> Test softmax nn: MNIST")
        
X_train, Y_train, X_test, Y_test, X_validation, Y_validation = mnist_data()

X_train = X_train[:, 0:5000]
Y_train = Y_train[:, 0:5000]

learning_rate = 0.001
# batch_size = 100
# display_step = 2
training_iteration = 7000
n = X_train.shape[0]
loss = nn.loss.categorical_crossentropy

relu = nna.relu()

X = nna.input(n)
X = nna.layer(50)(X)
X = relu(X)
X = nna.layer(20)(X)
X = relu(X)
X = nna.layer(10)(X)
X = nna.softmax()(X)

optimizer = nn.gradient_descent(learning_rate, training_iteration, loss)

model = nn.Model(X, optimizer)

print("fitting model....")

parameters, costs = model.fit(X_train, Y_train, True)

plot_cost(costs, learning_rate)

# train_acc = model.evaluate(X_train, Y_train, 'train')
# test_acc = model.evaluate(X_test, Y_test, 'test')
