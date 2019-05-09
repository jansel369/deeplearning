import torch as pt
from collections import namedtuple
from .stochastic import stochastic_optimization

Momentum = namedtuple('Momentum', 'loss, epochs, batch_size, learning_rate, beta1, optimize, param_update_f')

def vel_grad(grad, Vgrad, beta1):
    return beta1 * Vgrad + (1 - beta1) * grad

def vel_param_update(learning_rate, param, Vgrad):
    param -= learning_rate * Vgrad

    return param

def momentum_update_param_f(optimizer, to_avg):
    learning_rate = optimizer.learning_rate
    beta1 = optimizer.beta1
    V_grads = [0, 0]
    
    def update_param(dZ, param_grads, cache, parameters):
        current_cache, next_cache = cache
        A_prev, current_param = current_cache

        for i in range(len(param_grads)):
            V_grads[i] = vel_grad(param_grads[i], V_grads[i], beta1)
            current_param[i] = vel_param_update(learning_rate, current_param[i], V_grads[i])

        return dZ, None, cache, (current_param, parameters)
    
    return update_param

def momentum(loss, epochs, batch_size=64, learning_rate=0.009, beta1=0.9):

    optimizer = Momentum(loss, epochs, batch_size, learning_rate, beta1, stochastic_optimization, momentum_update_param_f)

    return optimizer