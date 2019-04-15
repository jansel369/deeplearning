import torch as pt

from .commons import *

def Input(size):
    layer = {
        "size": size
    }

    config = {
        "layers": [layer]
    }

    return config

def Layer(size, activation):
    def a(config):
        layer = {
            "size": size,
            "activation": activation
        }

        config["layers"].append(layer)

        return config

    return a

class Model():
    def __init__(self, config):
        self.config = config
    
    def optimization(self, optimizer):
        self.optimizer = optimizer

    def fit(self, X_train, Y_train):
        device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
        
        parameters = init_params(self.config["layers"], device)

        return self.optimizer(X_train, Y_train, parameters)
        
    # def evaluate(self, Y_train, Y_test):

    #     return "test"

