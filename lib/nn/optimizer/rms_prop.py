import torch as pt
from collections import namedtuple
from .stochastic import stochastic_optimization

RMSProp = namedtuple('RMSProp', 'loss, iterations, batch_size, learning_rate, beta2, epsilon, optimize, param_update_f')

def rms_grad(grad, Sgrad, beta2):
    return beta2 * Sgrad + (1 - beta2) * (grad ** 2)

def rms_param_update(learning_rate, param, grad, Sgrad, epsilon):
    param -= learning_rate * (grad / (Sgrad + epsilon).sqrt())

    return param

def rms_update_param_f(optimizer, to_avg):
    learning_rate = optimizer.learning_rate
    beta2 = optimizer.beta2
    epsilon = optimizer.epsilon
    S_grads = [0, 0]
    
    def update_param(dZ, param_grads, cache, parameters):
        current_cache, next_cache = cache
        A_prev, current_param = current_cache

        for i in range(len(param_grads)):
            S_grads[i] = rms_grad(param_grads[i], S_grads[i], beta2)
            current_param[i] = rms_param_update(learning_rate, current_param[i], param_grads[i], S_grads[i], epsilon)

        return dZ, None, cache, (current_param, parameters)
    
    return update_param

def rms_prop(loss, epochs, batch_size=64, learning_rate=0.001, beta2=0.9, epsilon=10e-8):

    optimizer = RMSProp(loss, epochs, batch_size, learning_rate, beta2, epsilon, stochastic_optimization, rms_update_param_f)

    return optimizer
