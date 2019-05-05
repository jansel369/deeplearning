import torch as pt

from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g

from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent

from . import commons as c

def vel_weight_f(beta):
    def calculate(VdW, dW):
        return beta * VdW + (1 - beta) * dW
    
    return calculate

def vel_bias_f(beta):
    def calculate(Vdb, db):
        return beta * Vdb + (1 - beta) * db

    return calculate
 
def std_update(beta):
    vel_weight = vel_weight_f(beta)
    vel_bias = vel_bias_f(beta)

    def moment_update(learning_rate, m):
        """ momentum standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        VdW = 0
        Vdb = 0

        def update(dZ, cache, parameters):
            """Momentum standard update
                cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
                parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
            """
            nonlocal VdW, Vdb # set flag to modify VdW, Vdb in closure

            current_cache, next_cache = cache
            A_prev, W, b = current_cache

            dW = weight_grad(dZ, A_prev)
            db = bias_grad(dZ)

            VdW = vel_weight(VdW, dW)
            Vdb = vel_bias(Vdb, db)

            W -= learning_rate * VdW
            b -= learning_rate * Vdb

            return dZ, cache, ((W, b), parameters)

        return update

    return moment_update

class Momentum(StochasticGradientDescent):
    def __init__(self, loss, epochs, batch_size=64, learning_rate=0.009, beta=0.9):
        super().__init__(loss, epochs, batch_size, learning_rate)
        self.beta = beta

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        update_dict = {
            "std_update": std_update(self.beta)
        }

        return super().optimize(X, Y, parameters, config, is_printable_cost, update_dict)
