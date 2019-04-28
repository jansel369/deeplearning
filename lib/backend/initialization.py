"""ref: https://www.kdnuggets.com/2018/06/deep-learning-best-practices-weight-initialization.html
"""

import torch as pt

glorot = 'glorot'
he = 'he'

def glorot_init(W, n, n_prev):
    return W * ((1 / n_prev) ** 0.5)

def he_init(W, n, n_prev):
    return W * ((2 / n_prev) ** 0.5)

init_dict = {
    glorot: glorot_init,
    he: he_init,
}