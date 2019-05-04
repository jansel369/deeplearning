from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
# from backend import propagation as p
# from backend import prediction as pred
from . import commons
from .CostEvaluator import CostEvaluator

from nn import propagation as p

""" constructing backward prop functions
    naming format: (activation + '_')<optimizer>_<type>_<method>
"""

def construct_backwards(layers, learnibng_rate, m):
    """ temporary constructor
        todo: batch norm
        backwards - array of backward functions
                  - 2 hidden, 1 output layers
                    - format: [update, liniar_grad, update, liniar_grad, update]
    """
    backwards = []
    backward_dict = { # for caching purposes to avoid creation redundancy
        'gr_std_update': gr_std_update(learnibng_rate, m),
    }

    backwards.append(backward_dict['gr_std_update'])

    for l in reversed(range(1, len(layers) - 1)):
        layer = layers[l]
        activation = layer['activation']

        # Add liniar grad calculation
        liniar_fn = activation + '_liniar_std_grad'

        if liniar_fn in backward_dict: # resuse cached function
            backwards.append(backward_dict[liniar_fn])
        else: # create & cache new function
            backward = liniar_std_grad(a.activation_backward_dict[activation])
            backwards.append(backward)

            backward_dict[liniar_fn] = backward

        # Add parameter update function
        update_fn = 'gr_std_update' # temporary
        backwards.append(backward_dict[update_fn])

    return backwards


""" parameters update
"""

def weight_std_grad(m):
    def f(dZ, A_prev):
        return (1 / m) * dZ.mm(A_prev.t())
    
    return f

def bias_std_grad(m):
    def f(dZ):
        return (1 / m) * dZ.sum(dim=1, keepdim=True)

    return f

def gr_std_update(learning_rate, m):

    weight_grad = weight_std_grad(m)
    bias_grad = bias_std_grad(m)

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

""" liniar grad
"""

def liniar_std_grad(activation_backward):
    def compute_liniar_std_grad(dZ, cache, parameters):
        current_cache, next_cache = cache
        A, W_preced, b_preced = current_cache
        
        dZ = W_preced.t().mm(dZ) * activation_backward(A)

        return dZ, next_cache, parameters

    return compute_liniar_std_grad

""" Backward propagation
"""

def backward_propagation(backwards, loss):
    grad_loss = g.loss_backward_dict[loss]
    
    def f(AL, Y, cache):
        parameters = None

        dZ = grad_loss(AL, Y)
        for backward in backwards:
            # print('running backward: ', backward.__name__)

            dZ, cache, parameters = backward(dZ, cache, parameters)

        return parameters

    return f

""" Optimizer
"""
class GradientDescent:

    def __init__(self, learning_rate, iterations, loss):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = loss

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        layers = config['layers']
        L = len(layers)
        m = Y.shape[1]

        cost_evaluator = CostEvaluator(self.loss, self.learning_rate, print_cost=True)

        forwards = config['forwards']
        backwards = construct_backwards(layers, self.learning_rate, m)

        forward_prop = p.forward_propagation(forwards, has_cache=True) # set has_cache=True for backprap training
        backward_prop = backward_propagation(backwards, self.loss)

        for i in range(self.iterations):

            AL, cache = forward_prop(X, parameters)

            parameters = backward_prop(AL, Y, cache)

            cost_evaluator.add_cost(i, AL, Y)

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