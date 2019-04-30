import copy
import torch as pt
from .commons import *
from .optimizer import *
from backend import cost as c
from backend import prediction as pred
from backend import parameters as params

class Model():
    def __init__(self, config):
        self._config = copy.deepcopy(config)

    def optimization(self, optimizer):
        self.optimizer = optimizer

    def fit(self, X_train, Y_train, is_printable_cost=False, device=get_device()):
        layers = self._config['layers']
        input_size = layers[0]['units']
        output_size = layers[-1]['units']
        n = X_train.shape[0]
        o = Y_train.shape[0]

        assert input_size == n, 'Invalid input size: ' + str(input_size) + ' : ' + str(n)
        assert output_size == o, 'Invalid output size: ' + str(output_size) + ' : ' + str(o)

        # device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
        # optimizer = self._config['optimization']['optimizer']

        parameters = params.init_params(self._config["layers"], device)

        parameters, costs = self.optimizer.optimize(X_train, Y_train, parameters, self._config, is_printable_cost)
        
        self._parameters = parameters

        return parameters, costs


    def evaluate(self, X, Y, name="evaluate"):
        accuracy, AL = predict(X, Y, self._parameters, self._config['layers'], self.optimizer.loss)
        cost = c.costs_dict[self.optimizer.loss](AL, Y)

        print(name)
        print("-> accuracy: %f" % (accuracy))
        print("-> cost: %f" %(cost))

        return accuracy, cost
    