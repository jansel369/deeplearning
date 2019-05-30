import torch as pt
import initialization as init


""" parameters format: ((W1, b1), ((W2, b2), ((gamma2, beta2), ((W3,b3), ...))))
"""

def _init_liniar_bn_params(n, n_prev, device, layer, parameters):
    if not layer.batch_norm:
        return parameters

    new_params = init.liniar_batch_norm_layers(n, n_prev, device)

    return (new_params, parameters)

def _liniar_params(layer, prev_layer, device, parameters):
    n = layer.units
    n_prev = prev_layer.units
    parameters = _init_liniar_bn_params(n, n_prev, device, layer, parameters)
        
    new_params = layer.initialization(n, n_prev, device)
    parameters = (new_params, parameters)

    return parameters

def _conv_prev_input(prev_layer):
    h, w, c = prev_layer.out_shape
    
    return h * w * c

def _init_conv_bn_params(f, n_C, n_prev, device, layer, parameters):
    if not layer.batch_norm:
        return parameters

    new_params = init.conv_bn_layers(f, n_C, n_prev, device)

    return (new_params, parameters)


def _conv_params(layer, prev_layer, device, parameters):
    n_prev = _conv_prev_input(prev_layer)
    f = layer.filters
    n_C = layer.channels
    n_C_prev = prev_layer.out_shape.channels

    parameters = _init_conv_bn_params(f, n_C, n_prev, device, layer, parameters)
    new_params = layer.initialization(f, n_C_prev, n_C, n_prev, device)
    parameters = (new_params, parameters)

    return parameters

def initialize_parameters(layers, device):
    parameters = None

    for l in reversed(range(1, len(layers))):
        layer = layers[l]
        prev_layer = layers[l-1]

        if type(layer).__name__ == 'LayerConfig':
            parameters = _liniar_params(layer, prev_layer, device, parameters)
        elif type(layer).__name__ == 'ConvLayer':
            parameters = _conv_params(layer, prev_layer, device, parameters)

    return parameters
