import torch as pt

from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g

from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent

from . import commons as c

# from backend import propagation as p
# from backend import prediction as pred

# def init_velocity(parameters):
#     velocity = {}

#     for param, value in parameters.items():
#         velocity["Vd"+param] = pt.zeros(value.shape, dtype=pt.double, device=value.device)

#     return velocity

# def update_velocity(grads, velocity, B):
#     for grad_k, grad_v in grads.items():
#         vel = "V" + grad_k
#         velocity[vel] = B * velocity[vel] + (1 - B) * grad_v

#     return velocity

# def update_parameters(L, parameters, velocity, learning_rate):
#     for l in range(1, L):

#         W_l = "W" + str(l)
#         b_l = "b" + str(l)

#         parameters[W_l] = parameters[W_l] - learning_rate * velocity["VdW" + str(l)]
#         parameters[b_l] = parameters[b_l] - learning_rate * velocity["Vdb" + str(l)]
    
#     return parameters

def std_update(beta):
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

            VdW = beta * VdW + (1 - beta) * dW
            Vdb = beta * Vdb + (1 - beta) * db

            W -= learning_rate * VdW
            b -= learning_rate * Vdb

            return dZ, cache, ((W, b), parameters)

        return update

    return moment_update

class Momentum(StochasticGradientDescent):
    def __init__(self, learning_rate, loss, epochs, batch_size, beta=0.9):
        super().__init__(learning_rate, loss, epochs, batch_size)
        self.beta = beta

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        update_dict = {
            "std_update": std_update(self.beta)
        }

        return super().optimize(X, Y, parameters, config, is_printable_cost, update_dict)
