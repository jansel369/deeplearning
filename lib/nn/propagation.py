
import torch as pt
from backend import gradient as g

"""cache format
    ((A_prev, W, b), ((A_prev, W, b),...))
"""

def liniar_forward(A_prev, params, has_cache, cache):
    current_params, next_params = params
    W, b = current_params

    Z = W.mm(A_prev) + b

    cache = ((A_prev, current_params), cache) if has_cache else None

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
    gamma_i = 1 / (var + epsilon).sqrt() # gamma identity
    Z_norm = mu_dev * gamma_i # Z normalized
    Z_tilda = gamma * Z_norm + beta # batch normalized

    cache = ((gamma, beta, mu, mu_dev, var, gamma_i, Z_norm, epsilon), cache) if has_cache else None

    return Z_tilda, next_params, cache

""" Activation forward
"""

def activation_forward_a(activation):
    def activation_forward(Z, params, has_cache, cache):
        A = activation(Z)

        return A, params, cache

    return activation_forward

""" Runs all forward propagations
"""

def forward_propagation(forwards, has_cache=False):
    def forward_prop(X, parameters):
        cache = None

        for forward in forwards:
            X, parameters, cache = forward(X, parameters, has_cache, cache)
        
        return X, cache

    return forward_prop

"""
     Back prop
"""

""" gradients
"""

def liniar_grad(activation_backward):
    def f(dA, A):
        return dA * activation_backward(A)

    return f

def activation_grad(W, dZ):
    return W.t().mm(dZ)

def update_param_a():
    def construct_update(optimizer, to_avg):
        return optimizer.param_update_f(optimizer, to_avg)

    return construct_update

def activation_grad_a(): # called from architecture
    def activation_grad_f2(optimizer, to_avg): # called from back prop initialization
        def activation_grad(dZ, param_grad, cache, parameters): # called during backprop
            current_cache, next_cache = cache
            A_prev, current_param = current_cache
            W, _ = current_param

            dA = W.t().mm(dZ)

            return dA, param_grad, cache, parameters
        return activation_grad
    return activation_grad_f2

def liniar_grad_f(activation_backward):
    def liniar_grad_f2(optimizer, to_avg):
        def liniar_grad(dA, param_grad, cache, parameters):
            current_cache, next_cache = cache
            A, preced_param = current_cache

            dZ = dA * activation_backward(A)
            
            return dZ, param_grad, next_cache, parameters
        return liniar_grad
    return liniar_grad_f2

""" Calculating gradient parameters dW, db
"""

def weight_grad(dZ, avg, A_prev):
    return avg * dZ.mm(A_prev.t())

def bias_grad(dZ, avg):
    return avg * dZ.sum(dim=1, keepdim=True)

def std_params_grad_f(dZ, A_prev, to_avg):
    dW = weight_grad(dZ, to_avg, A_prev)
    db = bias_grad(dZ, to_avg)

    return [dW, db]
    

def bn_prams_grad_f(dZ, A_prev, to_avg):
    dW = weight_grad(dZ, to_avg, A_prev)

    return [dW]

def param_grad_a(grad_calculator=std_params_grad_f): # calculates dW, db or dW only
    def param_grad_i(optimizer, to_avg):
        def calculate_param_grad(dZ, param_grad, cache, parameters):
            current_cache, next_cache = cache
            A_prev, current_param = current_cache

            param_grad = grad_calculator(dZ, A_prev, to_avg)

            return dZ, param_grad, cache, parameters
        
        return calculate_param_grad
    return param_grad_i

def batch_norm_grad_a(): # calculates dZ from batch norm backwarad
    def bn_grad_i(optimizer, to_avg):
        def bn_grad_backward(dZ_tilda, param_grad, cache, parameters):
            current_cache, next_cache = cache
            gamma, beta, mu, mu_dev, var, gamma_i, Z_norm, epsilon = current_cache

            dZ_norm = dZ_tilda * gamma
            dvar =  ( dZ_norm * mu_dev * (-0.5) * (gamma_i ** 3) ).sum(1, True)
            dmu =  (-dZ_norm * gamma_i).sum(1, True) + dvar * (-2 * to_avg) * mu_dev.sum(1, True)
            dZ = dZ_norm * gamma_i + (2 * to_avg) * dvar * mu_dev + to_avg * dmu
            
            return dZ, param_grad, next_cache, parameters
        
        return bn_grad_backward
    return bn_grad_i

def bn_param_grad_a(): # calculates dgamma, dbeta
    def param_grad(optimizer, to_avg):
        def bn_grad(dZ_tilda, param_grad, cache, parameters):
            current_cache, next_cache = cache
            gamma, beta, mu, mu_dev, var, gamma_i, Z_norm, epsilon = current_cache

            dgamma = to_avg * (dZ_tilda * Z_norm).sum(1, True)
            dbeta = to_avg * dZ_tilda.sum(1, True)

            return dZ_tilda, [dgamma, dbeta], cache, parameters
        
        return bn_grad
    return param_grad


""" Runs back prop
"""
def backward_propagation(backwards, loss):    
    def f(AL, Y, cache):
        parameters = None
        param_grad = None

        dZ = loss.grad_loss(AL, Y)

        for backward in reversed(backwards):
            dZ, param_grad, cache, parameters = backward(dZ, param_grad, cache, parameters)

        return parameters

    return f


""" Constructing backwards
"""
def construct_backwards(backwards, optimizer, to_avg):
    """ helper functions that returns array of backward propagation functions
        backwards - array of backward functions
                  - 2 hidden, 1 output layers
                    - format: [update, liniar_grad, (batch_norm_grad,) std/bn: update, liniar_grad, update]
    """

    new_backwards = []

    for i in range(1, len(backwards) - 1): #disinclude first and last backwards from list
        new_backwards.append(backwards[i](optimizer, to_avg))

    return backward_propagation(new_backwards, optimizer.loss)
