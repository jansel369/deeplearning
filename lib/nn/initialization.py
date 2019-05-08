import torch as pt

def weight_std(n, n_prev, device):
    return pt.randn(n, n_prev, dtype=pt.double, device=device)

def weight_he(n, n_prev, device):
    W = weight_std(n, n_prev, device)

    return W * ((2 / n_prev) ** 0.5)

def weight_glorot(n, n_prev, device):
    W = weight_std(n, n_prev, device)

    return W * ((1 / n_prev) ** 0.5)

def bias_std(n, n_prev, device):
    return pt.zeros(n, 1, dtype=pt.double, device=device)

def gamma_batch_norm_std(n, n_prev, device):
    return pt.ones(n, 1, dtype=pt.double, device=device)

def beta_batch_norm_std(n, n_prev, device):
    return bias_std(n, n_prev, device)


""" layers initialization
"""

def std_layers(n, n_prev, device):
    W = weight_std(n, n_prev, device)
    b = bias_std(n, n_prev, device)

def he_layers(n, n_prev, device):
    W = weight_he(n, n_prev, device)
    b = bias_std(n, n_prev, device)

    return [W, b]

def glorot_layers(n, n_prev, device):
    W = weight_glorot(n, n_prev, device)
    b = bias_std(n, n_prev, device)

    return [W, b]

def batch_norm_layers(n, n_prev, device):
    gamma = gamma_batch_norm_std(n, n_prev, device)
    beta = beta_batch_norm_std(n, n_prev, device)

    return [gamma, beta]

init_dict = {
    "std": std_layers,
    "he": he_layers,
    "glorot": glorot_layers,
}
