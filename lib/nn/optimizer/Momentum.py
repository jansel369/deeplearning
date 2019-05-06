import torch as pt

from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g

from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent

from . import commons as c
 
def vel_grad_f(beta):
    def calculate_vel_grad(Vgrad, grad):
        return beta * Vgrad + (1 - beta) * grad
    
    return calculate_vel_grad

def vel_param_update_f():
    def param_udpate(learning_rate, param, Vgrad):
        param -= learning_rate * Vgrad

        return param

    return param_udpate

def std_update(beta):
    vel_grad = vel_grad_f(beta)
    param_udpate = vel_param_update_f()

    def moment_update_f(learning_rate, m):
        """ momentum standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        VdW = 0
        Vdb = 0

        def momentum_update(dZ, cache, parameters):
            """Momentum standard update
                cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
                parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
            """
            nonlocal VdW, Vdb # set flag to modify VdW, Vdb in closure

            current_cache, next_cache = cache
            A_prev, W, b = current_cache

            dW = weight_grad(dZ, A_prev)
            db = bias_grad(dZ)

            VdW = vel_grad(VdW, dW)
            Vdb = vel_grad(Vdb, db)

            W = param_udpate(learning_rate, W, VdW)
            b = param_udpate(learning_rate, b, Vdb)

            return dZ, cache, ((W, b), parameters)

        return momentum_update

    return moment_update_f

def momentum_batch_norm_update_f(beta):
    vel_grad = vel_grad_f(beta)
    param_udpate = vel_param_update_f()

    def bn_update_f(learning_rate, m):
        """ batch norm
        """
        weight_grad = c.weight_std_grad(m)

        VdW = 0
        Vdgamma = 0
        Vdbeta = 0

        def bn_update(cache1, cache2, parameters):
            dZ, dgamma, dbeta, gamma, beta = cache1
            current_cache, next_cache = cache2
            A_prev, W, b = current_cache
            
            nonlocal VdW, Vdgamma, Vdbeta

            dW = weight_grad(dZ, A_prev)
            
            VdW = vel_grad(VdW, dW)
            Vdgamma = vel_grad(Vdgamma, dgamma)
            Vdbeta = vel_grad(Vdbeta, dbeta)

            W = param_udpate(learning_rate, W, VdW)
            gamma = param_udpate(learning_rate, gamma, Vdgamma)
            beta = param_udpate(learning_rate, beta, Vdbeta)

            parameters = ((W, b), ((gamma, beta), parameters))

            return dZ, cache2, parameters

        return bn_update

    return bn_update_f

class Momentum(StochasticGradientDescent):
    def __init__(self, loss, epochs, batch_size=64, learning_rate=0.009, beta=0.9):
        super().__init__(loss, epochs, batch_size, learning_rate)
        self.beta = beta

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        update_dict = {
            'std_update': std_update(self.beta),
            'bn_update': momentum_batch_norm_update_f(self.beta),
        }

        return super().optimize(X, Y, parameters, config, is_printable_cost, update_dict)
