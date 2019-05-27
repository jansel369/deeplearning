from backend import activation as a
import torch as pt
import copy
import propagation as prop
from nn import Config
from collections import namedtuple

ConvLayer = namedtuple('ConvLayer', 'filters, channels, padding, stride, activation, batch_norm')
PoolLayer = namedtuple('PoolLayer', 'type')

def update_conv_config(config, activation):
    layer = config.layers[-1]
    f, n_C, p, s, _, bn = layer
    config.layers[-1] = ConvLayer(f, n_C, p, s, activation, bn)

    return config

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
        
        config.backwards.append(prop.conv_backward_a())
        config.backwards.append(prop.update_param_a())
        config.backwards.append(prop.conv_param_grad_a())


def relu():
    def f(config):
        config = update_conv_config(config, a.relu)
        config.forwards.append(prop.activation_forward_a(a.relu.forward))
        config.backwards.appen(prop.conv_grad_a(a.relu.backward))

        return config
    return f

def max_pool(filters, strides):
    def f(config):
        pool_layer = PoolLayer('max')
        config.layers.append(pool_layer)
        config.forwards.append(prop.max_pool_forward_a(filters, strides))
        config.backwards.append(prop.max_pool_backward_a())

def avg_pool(filters, strides):
    def f(config):
        pool_layer = PoolLayer('avg')
        config.layers.append(pool_layer)
        config.forwards.append(prop.avg_pool_forward_a(filters, strides))
        config.backwards.append(prop.avg_pool_backward_a())

def flatten():
    def f(config):
        config.forwards.append(prop.flatten_forward)
        config.backwards.append(prop.flatten_backward_i)

# def batch_norm():
#     def f(config):
#         layer = config.layers[-1]
#         a, b, c, _, sequence = layer
#         sequence.append('batch_norm')

#         config.layers[-1] = LayerConfig(a, b, c, True, sequence)

#         config.forwards.append(prop.batch_norm_forward)

#         config.backwards[-1] = prop.liniar_param_grad_a(prop.bn_prams_grad_f)  # change param grad from dW, db to only dW
#         config.backwards.append(prop.batch_norm_grad_a()) # calculate dZ from batch norm
#         config.backwards.append(prop.update_param_a()) # update param gamma, beta
#         config.backwards.append(prop.bn_param_grad_a()) # calculate grad dgamma, dbeta

#         return config

#     return f

