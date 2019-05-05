import torch as pt


from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent

from . import commons as c
from .Momentum import vel_bias_f, vel_weight_f
from .RMSProp import prop_bias_f, prop_weight_f

def std_update(beta1, beta2, epsilon):
    vel_weight = vel_weight_f(beta1)
    vel_bias = vel_bias_f(beta1)
    prop_weight = prop_bias_f(beta2, epsilon)
    prop_bias = prop_bias_f(beta2, epsilon)

    def adam_update(learning_rate, m):
        """ adam standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        VdW = 0
        Vdb = 0
        SdW = 0
        Sdb = 0

        def update(dZ, cache, parameters):
            """Adam standard update
                cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
                parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
            """
            nonlocal SdW, Sdb, VdW, Vdb

            current_cache, next_cache = cache
            A_prev, W, b = current_cache

            dW = weight_grad(dZ, A_prev)
            db = bias_grad(dZ)

            SdW = prop_weight(SdW, dW)
            Sdb = prop_bias(Sdb, db)

            SdW = beta2 * SdW + (1 - beta2) * (dW ** 2)
            Sdb = beta2 * Sdb + (1 - beta2) * (db ** 2)

            W -= learning_rate * ( dW / (SdW + epsilon).sqrt() )
            b -= learning_rate * ( db / (Sdb + epsilon).sqrt() )

            return dZ, cache, ((W, b), parameters)

        return update

    return adam_update

class Adam(StochasticGradientDescent):
    def __init__(self, learning_rate, loss, epochs, batch_size, beta1=0.9, beta2=0.9, epsilon=10e-8):
        super().__init__(learning_rate, loss, epochs, batch_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        update_dict = {
            "std_update": std_update(self.beta1, self.beta2, self.epsilon)
        }

        return super().optimize(X, Y, parameters, config, is_printable_cost, update_dict)
