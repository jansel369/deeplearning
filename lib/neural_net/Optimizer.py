import copy

from .commons import *

def optimization(X, Y, parameters, learning_rate, iterations, loss):
    costs = []

    for i in range(iterations):

        has_cost = i % 100 == 0

        AL, caches = forward_propagation(X, parameters)
        if has_cost:
            cost = compute_cost(AL, Y)
            costs.append(cost)

            if is_printable_cost:
                print("Cost after iteration %i: %f percent" %(i, cost * 100))

        grads = backward_propagation(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters, costs

def GradientDescent(learning_rate, iterations, loss="cross_entropy"):
    def f(model):
        def optimizer(X, Y, parameters):
            return optimization(X, Y, parameters, learning_rate, iterations, loss)

        model.optimization(optimizer)

        return model

    return f