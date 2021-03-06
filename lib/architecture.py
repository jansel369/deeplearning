from .backend import activation as a
import torch as pt
import copy
from . import propagation as prop

from collections import namedtuple


""" Declare model configuration types
"""
LayerConfig = namedtuple('LayerConfig', 'units, activation, initialization, batch_norm, sequence')
ConvLayer = namedtuple('ConvLayer', 'filters, channels, padding, stride, activation, batch_norm')


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

def create_conv_layer_config(padding, stride, channels):
    return ConvLayer(padding, stride, channels, None, None, None, 'liniar', False)

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

"""Convolution functions
"""

def conv_input(img_height, img_width, img_channels):
    # layer = ConvLayer(img_height, img_width, img_channels, None, None, None, 'liniar', False)
    config = Config([], [], [])

    return config

def conv(filters, channels, padding, stride):
    def f(config):
        conv_layer = ConvLayer(filters, channels, padding, stride, 'liniar', False)
        config.layers.append(conv_layer)
        config.forwards.append(prop.conv_forward_a(padding, stride, channels))
        config.backwards.append(props.conv_backward_a())


def batch_norm():
    def f(config):
        layer = config.layers[-1]
        a, b, c, _, sequence = layer
        sequence.append('batch_norm')

        config.layers[-1] = LayerConfig(a, b, c, True, sequence)

        config.forwards.append(prop.batch_norm_forward)

        config.backwards[-1] = prop.liniar_param_grad_a(prop.bn_prams_grad_f)  # change param grad from dW, db to only dW
        config.backwards.append(prop.batch_norm_grad_a()) # calculate dZ from batch norm
        config.backwards.append(prop.update_param_a()) # update param gamma, beta
        config.backwards.append(prop.bn_param_grad_a()) # calculate grad dgamma, dbeta

        return config

    return f


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
