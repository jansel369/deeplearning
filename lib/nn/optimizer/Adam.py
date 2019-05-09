import torch as pt
from collections import namedtuple
from .momentum import vel_grad
from .rms_prop import rms_grad
from .stochastic import stochastic_optimization

Adam = namedtuple('Adam', 'loss, epochs, batch_size, learning_rate, beta1, beta2, epsilon, optimize, param_update_f')

def adam_param_update(learning_rate, param, vdc, sdc, epsilon):
    param -= learning_rate * (vdc / (sdc + epsilon).sqrt())

    return param

def adam_update_param_f(optimizer, to_avg):
    learning_rate = optimizer.learning_rate
    beta1 = optimizer.beta1
    beta2 = optimizer.beta2
    epsilon = optimizer.epsilon
    
    V_grads = [0, 0]
    S_grads = [0, 0]
    t = 1 # iteration, should start at 1 to avoid deviding by 0

    def update_param(dZ, param_grads, cache, parameters):
        nonlocal t

        current_cache, next_cache = cache
        A_prev, current_param = current_cache

        vcd = 1 / (1 - (beta1 ** t))
        scd = 1 / (1 - (beta2 ** t))

        for i in range(len(param_grads)):
            V_grads[i] = vel_grad(param_grads[i], V_grads[i], beta1)
            S_grads[i] = rms_grad(param_grads[i], S_grads[i], beta2)

            vdc = V_grads[i] * vcd
            sdc = S_grads[i] * scd

            current_param[i] = adam_param_update(learning_rate, current_param[i], vdc, sdc, epsilon)

        t += 1 # update t by 1 per iteration

        return dZ, None, cache, (current_param, parameters)
    
    return update_param

def adam(loss, epochs, batch_size=64, learning_rate=3e-4, beta1=0.9, beta2=0.999, epsilon=10e-8):

    optimizer = Adam(loss, epochs, batch_size, learning_rate, beta1, beta2, epsilon, stochastic_optimization, adam_update_param_f)

    return optimizer