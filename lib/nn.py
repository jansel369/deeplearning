import torch as pt
import copy

import propagation as prop
import backend.activation as a

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
        config.forwards.append(prop.liniar_forward)
        config.layers.append(layer)
        
        config.backwards.append(prop.liniar_backward_a()) # calculates dA
        config.backwards.append(prop.update_param_a()) # update parameter W, b
        config.backwards.append(prop.liniar_param_grad_a()) # calculate gradient dW, db

        return config

    return a

""" activations functions
"""

def relu(init='he'):
    def f(config):
        config = update_layer_config(config, a.relu, init)
        config.forwards.append(prop.activation_forward_a(a.relu.forward))
        config.backwards.append(prop.liniar_grad_f(a.relu.backward)) # calclulate grad dZ

        return config

    return f

def sigmoid(init='glorot'):
    def f(config):
        config = update_layer_config(config, a.sigmoid, init)
        config.forwards.append(prop.activation_forward_a(a.sigmoid.forward))
        config.backwards.append(prop.liniar_grad_f(a.sigmoid.backward)) # calclulate grad dZ

        return config
    return f

def softmax(init='glorot'):
    def f(config):
        config = update_layer_config(config, a.softmax, init)
        config.forwards.append(prop.activation_forward_a(a.softmax.forward))
        config.backwards.append(prop.liniar_grad_f(a.softmax.backward)) # calclulate grad dZ

        return config
    return f
