"""reference:
    https://arxiv.org/abs/1502.03167v1
    https://deepnotes.io/batchnorm
"""

def batch_norm_forward(Z, gamma, beta, ndim=1, epsion=1.001e-5):
    m = Z.shape[1]
    to_avg = 1 / m

    mu = to_avg * Z.sum(ndim, True)
    var = to_avg * ((Z - mu) ** 2).sum(ndim, True)
    Z_norm = (Z - mu) / ((var + epsion) ** 0.5)

    return = gamma * Z_norm + beta

def batch_norm_backward(dZ_tilda)