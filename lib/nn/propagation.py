
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


""" batch norm
    https://arxiv.org/abs/1502.03167v1
    https://deepnotes.io/batchnorm
"""

def batch_norm_forward(Z, params, has_cache, cache):
    epsilon=1.001e-5
    current_params, next_params = params
    gamma, beta = current_params
    m = Z.shape[1]
    to_avg = 1 / m

    mu = to_avg * Z.sum(1, True) # mean
    mu_dev = Z - mu # deviation from mean
    var = to_avg * (mu_dev ** 2).sum(1, True) # variance
    gamma_i = (var + epsilon).sqrt() # gamma identity
    Z_norm = mu_dev / gamma_i # Z normalized
    Z_tilda = gamma * Z_norm + beta # batch normalized

    cache = ((gamma, beta, mu, mu_dev, var, gamma_i, Z_norm, epsilon), cache) if has_cache else None

    return Z_tilda, next_params, cache

""" Activation forward
"""

def relu_forward(Z, params, has_cache, cache):
    Z[Z < 0] = 0.

    return Z, params, cache

def softmax_forward(Z, params, has_cache, cache):
    e = pt.exp(Z - Z.max(0)[0])
    relu = e / e.sum(0)

    return relu, params, cache

def sigmoid_forward(Z, params, has_cache, cache):
    sigmoid = 1 / (1 + pt.exp(-Z))

    return sigmoid, params, cache

""" gradients
"""

def liniar_grad(activation_backward):
    def f(dA, A):
        return dA * activation_backward(A)

    return f

def activation_grad(W, dZ):
    return W.t().mm(dZ)

""" Runs all forward propagations
"""

def forward_propagation(forwards, has_cache=False):
    def forward_prop(X, parameters):
        cache = None

        for forward in forwards:
            X, parameters, cache = forward(X, parameters, has_cache, cache)
        
        return X, cache

    return forward_prop
