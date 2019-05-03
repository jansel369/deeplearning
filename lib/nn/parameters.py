import torch as pt
from . import initialization as init

def initialization_batch_norm_parameters(n, n_prev, device, layer, parameters):
    if not layer['batch_norm']:
        return parameters

    new_params = init.batch_norm_layers(n, n_prev, device)

    return (new_params, parameters)

def initialize_parameters(layers, device):
    # layers = layers.copy
    parameters = None

    for l in reversed(range(1, len(layers))):
        layer = layers[l]
        n = layer['units']
        n_prev = layers[l-1]['units']

        new_params = init.init_dict[layer['initialization']](n, n_prev, device)

        parameters = (new_params, parameters)

        parameters = initialization_batch_norm_parameters(n, n_prev, device, layer, parameters)
    
    return parameters
