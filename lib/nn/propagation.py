
import torch as pt

"""cache format
    ((A_prev, W, b), ((A_prev, W, b),...))
"""

def liniar_forward(A_prev, params, has_cache, cache):
    current_params, next_params = params
    W, b = current_params

    Z = W.mm(A_prev) + b

    cache = ((A_prev, W, b), cache) if has_cache else None

    return Z, next_params, cache

def batch_norm_forward(Z, params, has_cache, cache):

    return Z

""" Activation forward
"""

def relu_forward(Z, params, has_cache, cache):
    Z[Z < 0] = 0.

    return Z, params, cache

def softmax_forward(Z, params, has_cache, cache):
    # print('Z', Z)
    # print('parmas', params)
    # print('has_cache; ', has_cache)
    # print('cache', cache)

    e = pt.exp(Z - Z.max(0)[0])

    return e / e.sum(0), params, cache

def sigmoid_forward(Z, params, has_cache, cache):
    return 1 / (1 + pt.exp(-Z)), params, cache

""" activation backward
"""

def relu_backward(A):
    # g'(z)
    g = (A > 0).double()
    return g

# def softmax_forward(Z, ndim=0):
#     # e = Z.exp()
#     # stable version
#     e = pt.exp(Z - Z.max(ndim)[0])

#     return e / e.sum(ndim)

def sigmoid_backward(A):
    # g'(z)
    g = A * (1 - A)
    return g

""" gradients
"""

def liniar_grad(activation_backward):
    def f(dA, A):
        return dA * activation_backward(A)

    return f

def activation_grad(W, dZ):
    return W.t().mm(dZ)
