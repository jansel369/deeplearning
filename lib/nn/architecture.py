import torch as pt
from backend import activation as a
import copy
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
        config.backwards.append(propagation.batch_norm_grad_a()) # calculate dZ from batch norm
        config.backwards.append(propagation.update_param_a()) # update param gamma, beta
        config.backwards.append(propagation.bn_param_grad_a()) # calculate grad dgamma, dbeta

        return config

    return f


""" activations functions
"""

def relu(init='he'):
    def f(config):
        config = update_layer_config(config, a.relu, init)
        config.forwards.append(propagation.activation_forward_a(a.relu.forward))
        config.backwards.append(propagation.liniar_grad_f(a.relu.backward)) # calclulate grad dZ

        return config

    return f

def sigmoid(init='glorot'):
    def f(config):
        config = update_layer_config(config, a.sigmoid, init)
        config.forwards.append(propagation.activation_forward_a(a.sigmoid.forward))
        config.backwards.append(propagation.liniar_grad_f(a.sigmoid.backward)) # calclulate grad dZ

        return config
    return f

def softmax(init='glorot'):
    def f(config):
        config = update_layer_config(config, a.softmax, init)
        config.forwards.append(propagation.activation_forward_a(a.softmax.forward))
        config.backwards.append(propagation.liniar_grad_f(a.softmax.backward)) # calclulate grad dZ

        return config
    return f
