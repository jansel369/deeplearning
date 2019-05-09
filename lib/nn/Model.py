import copy
import torch as pt
import time
import datetime

from .commons import *
from .optimizer import *

from . import parameters as params
from . import initialization as init
from . import propagation as p

class Model():
    def __init__(self, config, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.parameters = None

    def fit(self, X_train, Y_train, is_printable_cost=False, device=get_device()):
        layers = self.config.layers
        input_size = layers[0].units
        output_size = layers[-1].units
        n = X_train.shape[0]
        o = Y_train.shape[0]

        assert input_size == n, 'Invalid input size: ' + str(input_size) + ' : ' + str(n)
        assert output_size == o, 'Invalid output size: ' + str(output_size) + ' : ' + str(o)

        forwards = self.config.forwards
        backwards = self.config.backwards

        start_time = time.time() # trainig timer

        parameters = params.initialize_parameters(layers, device)
        parameters = self.optimizer.optimize(X_train, Y_train, parameters, self.optimizer, forwards, backwards, is_printable_cost)
        
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
    