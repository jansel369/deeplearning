import torch as pt

def liniar_forward(A_prev, params, has_cache):
    W, b = params

    Z = W.mm(A_prev) + b

    return Z

def batch_norm_forward(Z, params, has_cache):

    return Z

def relu_forward(Z, params, has_cache):
    Z[Z < 0] = 0.

    return Z

def softmax_forward(Z, params, has_cache):
    e = pt.exp(Z - Z.max(0)[0])

    return e / e.sum(0)

def sigmoid_forward(Z, params, has_cache):
    return 1 / (1 + pt.exp(-Z))