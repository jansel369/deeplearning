import torch as pt
from .commons import *
import copy

def create_layer_config(size):
    layer = {
        'size': size,
        'activations': []
    }

    return layer

def Input(size):
    layer = create_layer_config(size)

    config = {
        'layers': [layer]
    }

    return config

def Layer(size):
    def a(config):
        config = copy.deepcopy(config)

        layer = create_layer_config(size)

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
        

