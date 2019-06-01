
from collections import namedtuple
from propagation import construct_backwards, forward_propagation
import time

GradientDescent = namedtuple('GradientDescent', 'loss, iterations, learning_rate, optimize, param_update_f')


def gd_update_param_f(optimizer, to_avg):
    learning_rate = optimizer.learning_rate

    def update_param(dZ, param_grad, cache, parameters):
        current_cache, next_cache = cache
        A_prev, current_param = current_cache

        for i in range(len(param_grad)):
            current_param[i] -= learning_rate * param_grad[i]

        return dZ, None, cache, (current_param, parameters)
    
    return update_param

def gradient_optimization(X, Y, parameters, optimizer, forwards, backwards, print_cost=False, steps=100):
    costs = []
    iterations = optimizer.iterations
    m = Y.shape[0]
    to_avg = 1 / m
    
    compute_cost = optimizer.loss.compute_cost
    forward_prop = forward_propagation(forwards, True)
    back_prop = construct_backwards(backwards, optimizer, to_avg)

    for i in range(iterations):
        start_time = time.time()

        AL, cache = forward_prop(X, parameters)
        parameters = back_prop(AL, Y, cache)

        if i % steps == 0:
            cost_i = compute_cost(AL, Y)
            costs.append(cost_i)
            epoch_time = time.time() - start_time
            if print_cost:
                print("Cost after iteration %i: %f, time: %f" % (i + 1, cost_i, epoch_time))
            
            start_time = time.time()

    return parameters

def gradient_descent(loss, iterations, learning_rate=0.01):
    optimizer = GradientDescent(loss, iterations, learning_rate, gradient_optimization, gd_update_param_f)

    return optimizer
