from backend import activation as a
import torch as pt
import copy
import propagation as prop
from nn import Config
from collections import namedtuple
import initialization as init

OutShape = namedtuple('OutShape', 'height, width, channels')
ConvLayer = namedtuple('ConvLayer', 'out_shape, filters, channels, padding, stride, activation, initialization, batch_norm')
PoolLayer = namedtuple('PoolLayer', 'out_shape, pool, units')

def _update_conv_config(config, activation, init):
    layer = config.layers[-1]
    v, f, n_C, p, s, _, _, bn = layer
    config.layers[-1] = ConvLayer(v, f, n_C, p, s, activation, init, bn)

    return config

def _calculate_side(prev_side, filter, padding, stride):
    return int(((prev_side + 2 * padding - filter) / stride) + 1)

"""Convolution functions
"""

def conv_input(img_height, img_width, img_channels):
    out_shape = OutShape(img_height, img_width, img_channels)
    layer = ConvLayer(out_shape, None, None, None, None, None, None, False)
    config = Config([layer], [], [])

    return config

def conv(filters, channels, padding, stride):
    def f(config):
        prev_side = config.layers[-1].out_shape.height
        side = _calculate_side(prev_side, filters, padding, stride)
        out_shape = OutShape(side, side, channels)
        conv_layer = ConvLayer(out_shape, filters, channels, padding, stride, 'liniar', 'std', False)
        
        config.layers.append(conv_layer)
        
        config.forwards.append(prop.conv_forward_a(padding, stride, channels))
        
        config.backwards.append(prop.conv_backward_a())
        config.backwards.append(prop.update_param_a())
        config.backwards.append(prop.conv_param_grad_a())

        return config
    return f

def relu(initialization=init.conv_he_layers):
    def f(config):
        config = _update_conv_config(config, a.relu, initialization)
        config.forwards.append(prop.activation_forward_a(a.relu.forward))

        config.backwards.append(prop.conv_grad_a(a.relu.backward))

        return config
    return f

def _gen_out_shape(config, filters, stride):
        prev_out_shape = config.layers[-1].out_shape
        prev_side = prev_out_shape.height
        n_C_prev = prev_out_shape.channels
        side = _calculate_side(prev_side, filters, 0, stride)
        out_shape = OutShape(side, side, n_C_prev)
        units = side * side * n_C_prev

        return out_shape, units

def max_pool(filters, stride):
    def f(config):
        out_shape, units = _gen_out_shape(config, filters, stride)
        pool_layer = PoolLayer(out_shape, 'max', units)
        config.layers.append(pool_layer)
        config.forwards.append(prop.max_pool_forward_a(filters, stride))
        config.backwards.append(prop.max_pool_backward_a())

        return config
    return f

def avg_pool(filters, stride):
    def f(config):
        out_shape, units = _gen_out_shape(config, filters, stride)
        pool_layer = PoolLayer(out_shape, 'avg', units)
        config.layers.append(pool_layer)

        config.forwards.append(prop.avg_pool_forward_a(filters, stride))
       
        config.backwards.append(prop.avg_pool_backward_a())

        return config
    return f

def flatten():
    def f(config):
        config.forwards.append(prop.flatten_forward)
        config.backwards.append(prop.flatten_backward_i)

        return config
    return f

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

