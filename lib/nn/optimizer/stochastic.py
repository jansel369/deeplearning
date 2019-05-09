
from collections import namedtuple
from nn.propagation import construct_backwards, forward_propagation
from .gradient_descent import gd_update_param_f

GradientDescent = namedtuple('GradientDescent', 'loss, epochs, batch_size, learning_rate, optimize, param_update_f')

def stochastic_optimization(X, Y, parameters, optimizer, forwards, backwards, print_cost=False):
    costs = []
    opochs = optimizer.epochs
    batch_size = optimizer.batch_size
    to_avg = 1 / batch_size
    m = Y.shape[1]
    batch_count = int(m / batch_size)
    
    compute_cost = optimizer.loss.compute_cost
    forward_prop = forward_propagation(forwards, True)
    back_prop = construct_backwards(backwards, optimizer, to_avg)

    X_batches = X.split(batch_size, 1)
    Y_batches = Y.split(batch_size, 1)

    total_iterations = 0

    for i in range(opochs):
        for t in range(batch_count):
            total_iterations += 1

            X_b = X_batches[t]
            Y_b = Y_batches[t]

            AL, cache = forward_prop(X_b, parameters)
            parameters = back_prop(AL, Y_b, cache)

            if total_iterations % 100 == 0:
                cost = compute_cost(AL, Y_b)
                costs.append(cost)

                if print_cost:
                    print("Cost after epoch %i, batch %i: %f " % (i+1, t+1, cost))


    return parameters


def stochastic(loss, opochs, batch_size=64, learning_rate=0.01):
    optimizer = GradientDescent(loss, opochs, batch_size, learning_rate, stochastic_optimization, gd_update_param_f)

    return optimizer
