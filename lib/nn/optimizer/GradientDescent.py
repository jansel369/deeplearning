from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
from backend import propagation as p
from backend import prediction as pred
from . import commons

class GradientDescent:

    def __init__(self, learning_rate, iterations, loss):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = loss
    
    def optimize(self, X, Y, parameters, config, is_printable_cost):
        costs = []
        layers = config['layers']
        L = len(layers)

        compute_cost = c.costs_dict[self.loss]
        loss_backward = g.loss_backward_dict[self.loss]

        for i in range(self.iterations):

            has_cost = i % 100 == 0

            # print(i)

            AL, caches = p.forward_propagation(X, parameters, layers)

            if has_cost:
                cost = compute_cost(AL, Y)
                costs.append(cost)

                if is_printable_cost:
                    print("Cost after iteration %i: %f " %(i, cost))

            dZL = loss_backward(AL, Y)

            grads = p.backward_propagation(dZL, caches, layers)

            parameters = commons.update_parameters(L, parameters, grads, self.learning_rate)

        return parameters, costs
    