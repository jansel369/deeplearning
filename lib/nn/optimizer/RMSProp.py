import torch as pt

from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent

from . import commons as c

def prop_grad_f(beta2):
    def calculate_prop_grad(Sgrad, grad):
        return beta2 * Sgrad + (1 - beta2) * (grad ** 2)

    return calculate_prop_grad

def rms_param_update_f(epsilon):
    def param_update(learning_rate, param, grad, Sgrad):
        param -= learning_rate * (grad / (Sgrad + epsilon).sqrt())

        return param
    
    return param_update

def std_update(beta2, epsilon):
    prop_grad = prop_grad_f(beta2)
    param_update = rms_param_update_f(epsilon)

    def rms_update_f(learning_rate, m):
        """ rms standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        SdW = 0
        Sdb = 0

        def rms_update(dZ, cache, parameters):
            """Rms Prop standard update
                cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
                parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
            """
            nonlocal SdW, Sdb # set flag to modify SdW, VSb in closure

            current_cache, next_cache = cache
            A_prev, W, b = current_cache

            dW = weight_grad(dZ, A_prev)
            db = bias_grad(dZ)

            SdW = prop_grad(SdW, dW)
            Sdb = prop_grad(Sdb, db)

            W = param_update(learning_rate, W, dW, SdW)
            b = param_update(learning_rate, b, db, Sdb)

            return dZ, cache, ((W, b), parameters)

        return rms_update

    return rms_update_f

def rms_batch_norm_update_f(beta2, epsilon):
    prop_grad = prop_grad_f(beta2)
    param_update = rms_param_update_f(epsilon)

    def bn_update_f(learning_rate, m):
        """ batch norm
        """
        weight_grad = c.weight_std_grad(m)

        SdW = 0
        Sdgamma = 0
        Sdbeta = 0

        def bn_update(cache1, cache2, parameters):
            nonlocal SdW, Sdgamma, Sdbeta
            
            dZ, dgamma, dbeta, gamma, beta = cache1
            current_cache, next_cache = cache2
            A_prev, W, b = current_cache
            
            dW = weight_grad(dZ, A_prev)
            
            SdW = prop_grad(SdW, dW)
            Sdgamma = prop_grad(Sdgamma, dgamma)
            Sdbeta = prop_grad(Sdbeta, dbeta)

            W = param_update(learning_rate, W, dW, SdW)
            gamma = param_update(learning_rate, gamma, dgamma, Sdgamma)
            beta = param_update(learning_rate, beta, dgamma, Sdgamma)

            parameters = ((W, b), ((gamma, beta), parameters))

            return dZ, cache2, parameters

        return bn_update

    return bn_update_f

class RMSProp(StochasticGradientDescent):
    def __init__(self, loss, epochs, batch_size=64, learning_rate=0.001, beta2=0.9, epsilon=10e-8):
        super().__init__(loss, epochs, batch_size, learning_rate)
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        update_dict = {
            'std_update': std_update(self.beta2, self.epsilon),
            'bn_update': rms_batch_norm_update_f(self.beta2, self.epsilon),
        }

        return super().optimize(X, Y, parameters, config, is_printable_cost, update_dict)
