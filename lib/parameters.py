import torch as pt
import initialization as init


""" parameters format: ((W1, b1), ((W2, b2), ((gamma2, beta2), ((W3,b3), ...))))
"""

def initialization_batch_norm_parameters(n, n_prev, device, layer, parameters):
    if not layer.batch_norm:
        return parameters

    new_params = init.batch_norm_layers(n, n_prev, device)

    return (new_params, parameters)

def initialize_parameters(layers, device):
    parameters = None

    for l in reversed(range(1, len(layers))):
        layer = layers[l]
        n = layer.units
        n_prev = layers[l-1].units

        parameters = initialization_batch_norm_parameters(n, n_prev, device, layer, parameters)
        
        new_params = init.init_dict[layer.initialization](n, n_prev, device)
        parameters = (new_params, parameters)

    
    return parameters
