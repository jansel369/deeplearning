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
from nn.propagation import construct_backwards, forward_propagation

GradientDescent = namedtuple('GradientDescent', 'loss, iterations, learning_rate, optimize, param_update_f')

def update_param_f(optimizer):
    learning_rate = optimizer.learning_rate

    def update_param(dZ, param_grad, cache, parameters):
        current_cache, next_cache = cache
        A_prev, current_param = current_cache
        
        for i in range(len(param_grad)):
            current_param[i] -= learning_rate * param_grad[i]

        return dZ, None, (current_param, parameters)
    
    return update_param

def gradient_optimization(X, Y, parameters, optimizer, forwards, backwards, print_cost=False):
    costs = []
    iterations = optimizer.iterations
    m = Y.shape[1]
    to_avg = 1 / m
    
    compute_cost = optimizer.loss.compute_cost
    forward_prop = forward_propagation(forwards, True)
    back_prop = construct_backwards(backwards, optimizer, to_avg)

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
    m = iterations
    update_param_f = update_param_f
    optimizer = GradientDescent(loss, iterations, learning_rate, gradient_optimization)

    return optimizer

