import torch as pt

def liniar_weight_std(n, n_prev, device):
    return pt.randn(n_prev, n, dtype=pt.double, device=device)

def liniar_weight_he(n, n_prev, device):
    W = liniar_weight_std(n, n_prev, device)

    return W * ((2 / n_prev) ** 0.5)

def liniar_weight_glorot(n, n_prev, device):
    W = liniar_weight_std(n, n_prev, device)

    return W * ((1 / n_prev) ** 0.5)

def liniar_bias_std(n, n_prev, device):
    return pt.zeros(1, n, dtype=pt.double, device=device)

def liniar_gamma_batch_norm_std(n, n_prev, device):
    return pt.ones(1, n, dtype=pt.double, device=device)

def liniar_beta_batch_norm_std(n, n_prev, device):
    return liniar_bias_std(n, n_prev, device)


""" layers initialization
"""

def liniar_std_layers(n, n_prev, device):
    W = liniar_weight_std(n, n_prev, device)
    b = liniar_bias_std(n, n_prev, device)

def liniar_he_layers(n, n_prev, device):
    W = liniar_weight_he(n, n_prev, device)
    b = liniar_bias_std(n, n_prev, device)

    return [W, b]

def liniar_glorot_layers(n, n_prev, device):
    W = liniar_weight_glorot(n, n_prev, device)
    b = liniar_bias_std(n, n_prev, device)

    return [W, b]
 
def liniar_batch_norm_layers(n, n_prev, device):
    gamma = liniar_gamma_batch_norm_std(n, n_prev, device)
    beta = liniar_beta_batch_norm_std(n, n_prev, device)

    return [gamma, beta]

# init_dict = {
#     "std": liniar_std_layers,
#     "he": liniar_he_layers,
#     "glorot": liniar_glorot_layers,
# }

"""Convolution initializatoin
"""

def conv_weight_std(f, n_C, device):
    return pt.randn((f, f, n_C), dtype=pt.double, device=device)

def conv_bias_std(n_C, device):
    return pt.zeros((n_C, 1, 1, 1), dtype=pt.double, device=device)

def conv_weight_he(f, n_C, n_prev, device):
    return conv_weight_std(f, n_C, device) * ((2 / n_prev) ** 0.5)

def conv_weight_glorot(f, n_C, n_prev, device):
    return conv_weight_std(f, n_C, device) * ((1 / n_prev) ** 0.5)

def conv_gamma_bn_std(n_C, device):
    return pt.ones((n_C, 1, 1, 1), dtype=pt.double, device=device)

def conv_beta_bn_std(n_C, device):
    return conv_bias_std(n_C, device)

def conv_std_layers(f, n_C, n_prev, device):
    W = conv_weight_std(f, n_C, device)
    b = conv_bias_std(n_C, device)

    return [W, b]

def conv_he_layers(f, n_C, n_prev, device):
    W = conv_weight_he(f, n_C, n_prev, device)
    b = conv_bias_std(n_C, device)

    return [W, b]

def conv_glorot_layers(f, n_C, n_prev, device):
    W = conv_weight_glorot(f, n_C, n_prev, device)
    b = conv_bias_std(n_C, device)

    return [W, b]

def conv_bn_layers(f, n_C, n_prev, device):
    gamma = conv_gamma_bn_std(n_C, device)
    beta = conv_beta_bn_std(n_C, device)

    return [gamma, beta]
