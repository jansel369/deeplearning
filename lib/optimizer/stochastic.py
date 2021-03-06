import math
import time

from collections import namedtuple
from propagation import construct_backwards, forward_propagation
from .gradient_descent import gd_update_param_f

GradientDescent = namedtuple('GradientDescent', 'loss, epochs, batch_size, learning_rate, optimize, param_update_f')

def stochastic_optimization(X, Y, parameters, optimizer, forwards, backwards, print_cost=False, steps=100):
    costs = []
    opochs = optimizer.epochs
    batch_size = optimizer.batch_size
    to_avg = 1 / batch_size
    m = Y.shape[0]
    batch_count = math.ceil(m / batch_size)

    compute_cost = optimizer.loss.compute_cost
    forward_prop = forward_propagation(forwards, True)
    back_prop = construct_backwards(backwards, optimizer, to_avg)

    X_batches = X.split(batch_size, 0)
    Y_batches = Y.split(batch_size, 0)

    total_iterations = 0

    print("batch count: ", batch_count)

    for i in range(opochs):
        start_time = time.time()

        for t in range(batch_count):
            total_iterations += 1

            X_b = X_batches[t]
            Y_b = Y_batches[t]

            AL, cache = forward_prop(X_b, parameters)
            parameters = back_prop(AL, Y_b, cache)

            if total_iterations % steps == 0:
                cost = compute_cost(AL, Y_b)
                costs.append(cost)

                if print_cost:
                    step_time = time.time() - start_time
                    print("Cost after (time: %f, epoch %i, batch %i): %f " % (step_time, i+1, t+1, cost))
                    start_time = time.time()

    return parameters


def stochastic(loss, opochs, batch_size=64, learning_rate=0.01):
    optimizer = GradientDescent(loss, opochs, batch_size, learning_rate, stochastic_optimization, gd_update_param_f)

    return optimizer
