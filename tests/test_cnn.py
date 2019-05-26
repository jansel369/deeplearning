import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../datasets")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../utils")

# import  nn
import contrib
# from nn import architecture as nna
import torch as pt
import unittest
from loader import *
from data_divider import *
from plot_cost import *
import matplotlib.pyplot as plt

import mnist.input_data as data_source


# def mnist_data():
mnist = data_source.read_data_sets("datasets/mnist/data/", reshape=False, one_hot=True)
device = contrib.get_device()

X_train = contrib.from_numpy(mnist.train.images, device)#[:, 0:3000]
Y_train = contrib.from_numpy(mnist.train.labels, device)#[:, 0:3000]

X_test = contrib.from_numpy(mnist.test.images, device)
Y_test = contrib.from_numpy(mnist.test.labels, device)

X_validation = contrib.from_numpy(mnist.validation.images, device)
Y_validation = contrib.from_numpy(mnist.validation.labels, device)

print("\ntrain shape: X=%s, Y=%s" % (X_train.shape, Y_train.shape))
print("validation shape: X=%s, Y=%s" % (X_validation.shape, Y_validation.shape))
print("test shape: X=%s, Y=%s" % (X_test.shape, Y_test.shape))


