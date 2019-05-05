import torch as pt

from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent

from . import commons as c

def prop_weight_f(beta2, epsion):
    def calculate(SdW, dW):
        return beta2 * SdW + (1 - beta2) * (dW ** 2)
    
    return calculate

def prop_bias_f(beta2, epsilon):
    def calculate(Sdb, db):
        return beta2 * Sdb + (1 - beta2) * (db ** 2)
    
    return calculate

def std_update(beta2, epsilon):
    prop_weight = prop_bias_f(beta2, epsilon)
    prop_bias = prop_bias_f(beta2, epsilon)

    def rms_update(learning_rate, m):
        """ rms standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        SdW = 0
        Sdb = 0

        def update(dZ, cache, parameters):
            """Rms Prop standard update
                cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
                parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
            """
            nonlocal SdW, Sdb # set flag to modify SdW, VSb in closure

            current_cache, next_cache = cache
            A_prev, W, b = current_cache

            dW = weight_grad(dZ, A_prev)
            db = bias_grad(dZ)

            SdW = prop_weight(SdW, dW)
            Sdb = prop_bias(Sdb, db)

            W -= learning_rate * ( dW / (SdW + epsilon).sqrt() )
            b -= learning_rate * ( db / (Sdb + epsilon).sqrt() )

            return dZ, cache, ((W, b), parameters)

        return update

    return rms_update

class RMSProp(StochasticGradientDescent):
    def __init__(self, learning_rate, loss, epochs, batch_size, beta2=0.9, epsilon=10e-8):
        super().__init__(learning_rate, loss, epochs, batch_size)
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        update_dict = {
            "std_update": std_update(self.beta2, self.epsilon)
        }

        return super().optimize(X, Y, parameters, config, is_printable_cost, update_dict)
