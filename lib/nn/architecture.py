import torch as pt
from backend import activation as a
from backend import initialization as init
from backend import parameters
# from .commons import *
import copy
# import optimizer
from . import propagation

from collections import namedtuple


""" Declare model configuration types
"""

Config = namedtuple('Config', 'layers, forwards')
LayerConfig =namedtuple('LayerConfig', 'units, activation, initialization, batch_norm, sequence')

""" Helper functions
"""

def create_layer_config(units, activation='linear'):
    layer = LayerConfig(units, activation, 'std', False, ['liniar'])

    return layer

def update_layer_config(config, activation, init):
    layer = config.layers[-1]

    units, _, _, batch_norm, sequence = layer
    sequence.append(activation)

    config.layers[-1] = LayerConfig(units, activation, init, batch_norm, sequence)

    return config

""" liniar functions
"""

def input(units):
    layer = LayerConfig(units, 'liniar', 'std', False, ['input'])
    config = Config([layer], [])

    return config

def layer(units):
    def a(config):
        layer = create_layer_config(units)
        config.forwards.append(propagation.liniar_forward)

        config.layers.append(layer)

        return config

    return a

""" activations functions
"""

def relu(init=init.he):
    def f(config):
        config = update_layer_config(config, a.relu, init)
        config.forwards.append(propagation.relu_forward)

        return config

    return f

def sigmoid(init=init.glorot):
    def f(config):
        config = update_layer_config(config, a.sigmoid, init)
        config.forwards.append(propagation.sigmoid_forward)

        return config
    return f

def softmax(init=init.glorot):
    def f(config):
        config = update_layer_config(config, a.softmax, init)
        config.forwards.append(propagation.softmax_forward)

        return config
    return f

def batch_norm():
    def f(config):
        config = copy.deepcopy(config)
        
        layer = config.layers[-1]
        layer.batch_norm = True
        layer.sequence.append('batch_norm')

        return config

    return f


# def leaky_relu(init=init.he):
#     def f(config):
#         return update_layer_config(config, a.leaky_relu, init)

#     return f



# def tanh(init=init.glorot):
#     def f(config):
#         return update_layer_config(config, a.tanh, init)

#     return f

# def dense(units, activation, init=''):
#     def f(config):
#         config = copy.deepcopy(config)
#         lc = create_layer_config(units, activation)
#         if init == '':
#             lc['initialization'] = init_by_activation[activation]
#             lc['parameter_type'] = parameters.standard

#         config['layers'].append(lc)

#         return config
    
#     return f
