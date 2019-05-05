import torch as pt

from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
# from backend import propagation as p
# from backend import prediction as pred

from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent

from . import commons as c

# def init_rms(parameters):
#     rms = {}

#     for param, value in parameters.items():
#         rms["Sd"+param] = pt.zeros(value.shape, dtype=pt.double, device=value.device)

#     return rms

# def update_rms(grads, rms, B):
#     for grad_k, grad_v in grads.items():
#         vel = "S" + grad_k
#         rms[vel] = B * rms[vel] + (1 - B) * (grad_v ** 2)

#     return rms

# def update_parameters(L, parameters, grads, rms, learning_rate, epsilon):
#     for l in range(1, L):
#         l_s = str(l)

#         parameters["W"+l_s] -= ( learning_rate *  grads["dW"+l_s] / (rms["SdW"+l_s] + epsilon).sqrt() )
#         parameters["b"+l_s] -= ( learning_rate * grads["db"+l_s] / (rms["Sdb"+l_s] + epsilon).sqrt() ) 
    
#     return parameters

def std_update(beta2, epsilon):
    def moment_update(learning_rate, m):
        """ momentum standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        SdW = 0
        Sdb = 0

        def update(dZ, cache, parameters):
            """Momentum standard update
                cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
                parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
            """
            nonlocal SdW, Sdb # set flag to modify SdW, VSb in closure

            current_cache, next_cache = cache
            A_prev, W, b = current_cache

            dW = weight_grad(dZ, A_prev)
            db = bias_grad(dZ)

            SdW = beta2 * SdW + (1 - beta2) * (dW ** 2)
            Sdb = beta2 * Sdb + (1 - beta2) * (db ** 2)

            W -= learning_rate * ( dW / (SdW + epsilon).sqrt() )
            b -= learning_rate * ( db / (Sdb + epsilon).sqrt() )

            return dZ, cache, ((W, b), parameters)

        return update

    return moment_update

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
