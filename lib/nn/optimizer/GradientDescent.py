from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
# from backend import propagation as p
# from backend import prediction as pred
from . import commons
from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from . import commons as c

def gd_std_update(learning_rate, m):
    """ gradient descent standard update
    """

    weight_grad = c.weight_std_grad(m)
    bias_grad = c.bias_std_grad(m)

    def update(dZ, cache, parameters):
        """Gradient Descent standard update
            cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
            parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
        """
        current_cache, next_cache = cache
        A_prev, W, b = current_cache

        dW = weight_grad(dZ, A_prev)
        db = bias_grad(dZ)

        W -= learning_rate * dW
        b -= learning_rate * db

        return dZ, cache, ((W, b), parameters)

    return update

""" activatoin grad
"""

# def activation_grad(dZ, cache, parameters):
#     current_cache, next_cache = cache
#     A_prev, W, b = current_cache

#     dA = W.t().mm(dZ)

#     return dA, cache, parameters

""" Optimizer
"""

update_dict = {
    'std_update': gd_std_update,
}

class GradientDescent:

    def __init__(self, learning_rate, loss, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = loss

    def optimize(self, X, Y, parameters, config, is_printable_cost, update_dict=update_dict):
        layers = config.layers
        L = len(layers)
        m = Y.shape[1]

        cost_evaluator = CostEvaluator(self.loss, self.learning_rate, print_cost=True)

        forwards = config.forwards
        backwards = c.construct_backwards(update_dict, layers, self.learning_rate, m)

        forward_prop = p.forward_propagation(forwards, has_cache=True) # set has_cache=True for backprap training
        backward_prop = c.backward_propagation(backwards, self.loss)

        for i in range(self.iterations):

            AL, cache = forward_prop(X, parameters)

            parameters = backward_prop(AL, Y, cache)

            cost_evaluator.batch_cost(i, AL, Y)

        return parameters, cost_evaluator

# def f_prop_cache_debugger(cache):
#     # curr_c, next_c = cache
#     counter = 1

#     while cache != None:
#         curr_c, next_c = cache
#         A_p, W, b = curr_c
#         print(counter, A_p.shape, W.shape, b.shape)
#         counter += 1

#         cache = next_c

# def b_prop_params_debugger(cache):
# # curr_c, next_c = cache
#     counter = 1

#     while cache != None:
#         curr_c, next_c = cache
#         W, b = curr_c
#         print(counter,  W.shape, b.shape)
#         counter += 1

#         cache = next_c