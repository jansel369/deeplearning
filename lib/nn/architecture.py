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

Config = namedtuple('Config', 'layers, forwards, backwards')
LayerConfig = namedtuple('LayerConfig', 'units, activation, initialization, batch_norm, sequence')

""" Helper functions
"""

def create_layer_config(units, activation='linear'):
    layer = LayerConfig(units, activation, 'std', False, ['liniar'])

    return layer

def update_layer_config(config, activation, init):
    layer = config.layers[-1]

    a, _, _, d, sequence = layer
    sequence.append(activation)

    config.layers[-1] = LayerConfig(a, activation, init, d, sequence)

    return config

""" liniar functions
"""

def input(units):
    layer = LayerConfig(units, 'liniar', 'std', False, ['input'])
    config = Config([layer], [], [])

    return config

def layer(units):
    def a(config):
        layer = create_layer_config(units)
        config.forwards.append(propagation.liniar_forward)
        config.layers.append(layer)
        
        config.backwards.append(propagation.activation_grad_a()) # calculates dA
        config.backwards.append(propagation.update_param_a()) # update parameter W, b
        config.backwards.append(propagation.param_grad_a()) # calculate gradient dW, db

        return config

    return a

def batch_norm():
    def f(config):
        layer = config.layers[-1]
        a, b, c, _, sequence = layer
        sequence.append('batch_norm')

        config.layers[-1] = LayerConfig(a, b, c, True, sequence)

        config.forwards.append(propagation.batch_norm_forward)

        config.backwards[-1] = propagation.param_grad_a(propagation.bn_prams_grad_f)  # change param grad from dW, db to only dW
        config.backend.append(propagation.batch_norm_grad_a) # calculate dZ from batch norm
        config.backend.append(propagation.update_param_a()) # update param gamma, beta
        config.backend.append(propagation.bn_param_grad_a()) # calculate grad dgamma, dbeta

        return config

    return f


""" activations functions
"""

def relu(init=init.he):
    def f(config):
        config = update_layer_config(config, a.relu, init)
        config.forwards.append(propagation.relu_forward)
        config.forwards.append(propagation.liniar_grad_f(a.relu_backward)) # calclulate grad dZ

        return config

    return f

def sigmoid(init=init.glorot):
    def f(config):
        config = update_layer_config(config, a.sigmoid, init)
        config.forwards.append(propagation.sigmoid_forward)
        config.forwards.append(propagation.liniar_grad_f(a.sigmoid_backward)) # calclulate grad dZ

        return config
    return f

def softmax(init=init.glorot):
    def f(config):
        config = update_layer_config(config, a.softmax, init)
        config.forwards.append(propagation.softmax_forward)
        config.forwards.append(propagation.liniar_grad_f(a.softmax_backward)) # calclulate grad dZ

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

