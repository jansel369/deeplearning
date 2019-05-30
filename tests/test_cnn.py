import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../datasets")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../utils")

import  nn
import cnn
import contrib

import torch as pt
import unittest

from loader import *
from data_divider import *
from plot_cost import *
import matplotlib.pyplot as plt

import mnist.input_data as data_source
import seed_cnn as seed

# mnist = data_source.read_data_sets("datasets/mnist/data/", reshape=False, one_hot=True)
device = contrib.get_device()

# X_train = contrib.from_numpy(mnist.train.images, device)#[:, 0:3000]
# Y_train = contrib.from_numpy(mnist.train.labels, device)#[:, 0:3000]

# X_test = contrib.from_numpy(mnist.test.images, device)
# Y_test = contrib.from_numpy(mnist.test.labels, device)

# X_validation = contrib.from_numpy(mnist.validation.images, device)
# Y_validation = contrib.from_numpy(mnist.validation.labels, device)

# print("\ntrain shape: X=%s, Y=%s" % (X_train.shape, Y_train.shape))
# print("validation shape: X=%s, Y=%s" % (X_validation.shape, Y_validation.shape))
# print("test shape: X=%s, Y=%s" % (X_test.shape, Y_test.shape))

# # print(X_train[0])
# # print(Y_train[0])

# f = open('x.txt', 'w')
# f.write(str(X_train[1].tolist()))
# f.write('\n\n')
# f.write(str(Y_train[1].tolist()))
# f.close()


X_train = seed.X_train
Y_train = seed.Y_train
device = seed.device

print("\ntrain shape: X=%s, Y=%s" % (X_train.shape, Y_train.shape))



# testing lenet-5 architecture
img_height = X_train.shape[1]
img_width = X_train.shape[2]
img_channels = X_train.shape[3]

X = cnn.conv_input(img_height, img_width, img_channels)
X = cnn.conv(5, 6, 0, 1)(X)
X = cnn.relu()(X)
X = cnn.avg_pool(2, 2)(X)
X = cnn.conv(5, 16, 0, 1)(X)
X = cnn.relu()(X)
X = cnn.max_pool(2, 2)(X)
X = cnn.flatten()(X)
X = nn.layer(120)(X)
X = nn.relu()(X)
X = nn.layer(10)(X)
X = nn.softmax()(X)

loss = contrib.loss.categorical_crossentropy

# adam = contrib.adam(loss, 7)
gd = contrib.gradient_descent(loss, 500)

model = contrib.Model(X, gd)

parameters, cost_evaluator = model.fit(X_train, Y_train, print_cost=True)

train_acc = model.evaluate(X_train, Y_train, 'train')
# test_acc = model.evaluate(X_test, Y_test, 'test')
