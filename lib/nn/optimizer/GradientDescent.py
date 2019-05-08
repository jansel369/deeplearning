from backend import loss as l
from backend import activation as a
from backend import cost
from backend import gradient as g
# from backend import propagation as p
# from backend import prediction as pred
from . import commons
from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from . import commons as c
from collections import namedtuple

GradientDescent = namedtuple('GradientDescent', 'loss, iterations, learning_rate, optimize')

def update_param_f(hyp_params):
    learning_rate = hyp_params.learning_rate

    def update_param(dZ, param_grad, cache, parameters):
        current_cache, next_cache = cache
        A_prev, current_param = current_cache
        
        for i in range(len(param_grad)):
            current_param[i] -= learning_rate * param_grad[i]

        return dZ, None, (current_param, parameters)
    
    return update_param

def gradient_optimization(X, Y, parameters, hyper_params, forward_prop, back_prop, print_cost=False):
    costs = []
    m = Y.shape[1]
    iterations = hyper_params.iterations
    compute_cost = cost.costs_dict[hyper_params.loss]

    for i in range(iterations):
        AL, cache = forward_prop(X, parameters)
        parameters = back_prop(AL, Y, cache)

        if i % 100 == 0:
            cost_i = compute_cost(AL, Y)
            costs.append(cost_i)
            if print_cost:
                print("Cost after iteration %i: %f " % (i + 1, cost_i))

    return parameters

def gradient_descent(loss, iterations, learning_rate=0.01):
    optimizer = GradientDescent(loss, iterations, learning_rate, gradient_optimization)

    return optimizer

