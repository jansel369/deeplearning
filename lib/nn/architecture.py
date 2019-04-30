import torch as pt
from backend import activation as a
from backend import initialization as init
from backend import parameters
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

def update_layer_config(config, activation, init):
    config = copy.deepcopy(config)

    config['layers'][-1]['activation'] = activation
    config['layers'][-1]['initialization'] = init

    return config


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
        return update_layer_config(config, a.relu, init)

    return f

def leaky_relu(init=init.he):
    def f(config):
        return update_layer_config(config, a.leaky_relu, init)

    return f

def sigmoid(init=init.glorot):
    def f(config):
        return update_layer_config(config, a.sigmoid, init)

    return f

def tanh(init=init.glorot):
    def f(config):
        return update_layer_config(config, a.tanh, init)

    return f

def softmax(init=init.glorot):
    def f(config):
        return update_layer_config(config, a.softmax, init)
    
    return f

def dense(units, activation, init=''):
    def f(config):
        config = copy.deepcopy(config)
        lc = create_layer_config(units, activation)
        if init == '':
            lc['initialization'] = init_by_activation[activation]
            lc['parameter_type'] = parameters.standard

        config['layers'].append(lc)

        return config
    
    return f

def batch_norm():
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['parameter_type'] = parameters.batch_norm

        return config

    return f
