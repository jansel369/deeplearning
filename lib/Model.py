import copy
import torch as pt
import time
import datetime

from commons import *
from optimizer import *

import parameters as params
import initialization as init
import propagation as p

class Model():
    def __init__(self, config, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.parameters = None

    def fit(self, X_train, Y_train, print_cost=False, device=get_device()):
        layers = self.config.layers
        # n_0 = layers[0].units
        # n_L = layers[-1].units
        # n_0a = X_train.shape[1]
        # n_La = Y_train.shape[1]

        # assert n_0 == n_0a, 'Invalid input size: ' + str(n_0) + ' & ' + str(n_0a)
        # assert n_L == n_La, 'Invalid output size: ' + str(n_L) + ' & ' + str(n_La)

        forwards = self.config.forwards
        backwards = self.config.backwards

        start_time = time.time() # trainig timer

        parameters = params.initialize_parameters(layers, device)
        parameters = self.optimizer.optimize(X_train, Y_train, parameters, self.optimizer, forwards, backwards, print_cost)
        
        self.parameters = parameters

        end_time = time.time()
        print("\nTraining time: ", datetime.timedelta(seconds=end_time - start_time), '\n')

        return parameters


    def evaluate(self, X, Y, name="evaluate"):
        loss = self.optimizer.loss

        AL, _ = p.forward_propagation(self.config.forwards)(X, self.parameters)

        accuracy = loss.pred_acc(AL, Y)
        cost = loss.compute_cost(AL, Y)

        print(name)
        print("-> accuracy: %f" % (accuracy))
        print("-> cost: %f" %(cost))

        return accuracy, cost
    