import copy
import torch as pt
from collections import namedtuple
import time
import datetime

from .commons import *
from .optimizer import *
from backend import cost as c
# from backend import prediction as pred

from backend import accuracy as pred_acc

from . import parameters as params
from . import initialization as init
from . import propagation as p

# _Model = namedtuple('Model', 'fit, evaluate')

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
        parameters, cost_evaluator = self.optimizer.optimize(X_train, Y_train, parameters, self.optimizer, forwards, backwards, is_printable_cost)
        
        self.parameters = parameters

        end_time = time.time()
        print("Training time: ", datetime.timedelta(seconds=end_time - start_time))

        return parameters, cost_evaluator


    def evaluate(self, X, Y, name="evaluate"):
        loss = self.optimizer.loss

        AL, _ = p.forward_propagation(self.config.forwards)(X, self.parameters)

        accuracy = loss.pred_acc(AL, Y)
        cost = loss.compute_cost(AL, Y)

        print(name)
        print("-> accuracy: %f" % (accuracy))
        print("-> cost: %f" %(cost))

        return accuracy, cost
    


# class Model():
#     def __init__(self, config, optimizer):
#         self._config = copy.deepcopy(config)
#         self.optimizer = optimizer

    # def optimization(self, optimizer):
    #     self.optimizer = optimizer

    