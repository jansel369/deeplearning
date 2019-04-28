import torch as pt
from backend import activation as a
from backend import initialization as init
# from .commons import *
import copy
# import optimizer

init_by_activation = {
    a.relu: init.he,
    a.leaky_relu: init.he,
    a.softmax: init.glorot,
    a.sigmoid: init.glorot,
    a.tanh: init.glorot
}

def create_layer_config(units, activation=''):
    layer = {
        'units': units,
        'activation': activation,
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

def relu(init=init.he):
    def f(config):
        config = copy.deepcopy(config)

        config['layers'][-1]['activation'] = a.relu
        config['layers'][-1]['initialization'] = init

        return config

    return f

def leaky_relu(init=init.he):
    def f(config):
        config = copy.deepcopy(config)

        config['layers'][-1]['activation'] = a.leaky_relu
        config['layers'][-1]['initialization'] = init

        return config

    return f

def sigmoid(init=init.glorot):
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activation'] = a.sigmoid
        config['layers'][-1]['initialization'] = init

        return config

    return f

def tanh(init=init.glorot):
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activation'] = a.tanh
        config['layers'][-1]['initialization'] = init

        return config

    return f

def softmax(init=init.glorot):
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activation'] = a.softmax
        config['layers'][-1]['initialization'] = init
        
        return config
    
    return f

def dense(units, activation, init=''):
    def f(config):
        config = copy.deepcopy(config)
        lc = create_layer_config(units, activation)
        if init == '':
            lc['initialization'] = init_by_activation[activation]

        config['layers'].append(lc)

        return config
    
    return f
