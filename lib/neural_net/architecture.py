import torch as pt
import core.activation as a
# from .commons import *
import copy
# import optimizer

def create_layer_config(size):
    layer = {
        'size': size,
        'activation': '',
    }

    return layer

def input(size):
    layer = create_layer_config(size)

    config = {
        'layers': [layer]
    }

    return config

def layer(size):
    def a(config):
        config = copy.deepcopy(config)

        layer = create_layer_config(size)

        config["layers"].append(layer)

        return config

    return a

def relu():
    def f(config):
        config = copy.deepcopy(config)

        config['layers'][-1]['activation'] = a.relu

        return config

    return f

def rigmoid():
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activation'] = a.sigmoid

        return config

    return f

def softmax():
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activation'] = a.softmax
        
        return config
    
    return f
