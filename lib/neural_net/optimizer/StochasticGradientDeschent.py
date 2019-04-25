from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
from backend import propagation as p
from backend import prediction as pred

from . import commons

class StochasticGradientDescent:
    def __init__(self, learning_rate, iterations, batch_size, loss):
        self.learning_rate = learning_rate
        self.epochs = iterations
        self.batch_size = batch_size
        self.loss = loss
    
    def optimize(self, X, Y, parameters, config, is_printable_cost):
        costs = []
        layers = config['layers']
        L = len(layers)
        m = Y.shape[1 ]

        compute_cost = c.costs_dict[self.loss]
        loss_backward = g.loss_backward_dict[self.loss]

        batch_iterations = int(m / self.batch_size)
        count = 0

        for i in range(self.epochs):

            for t in range(batch_iterations):
                batch_start = t * self.batch_size
                batch_end = batch_start + self.batch_size

                X_t = X[:, batch_start:batch_end]
                Y_t = Y[:, batch_start:batch_end]

                # print(X_t.shape)
                # print(Y_t.shape)

                AL, caches = p.forward_propagation(X_t, parameters, layers)
                # print(AL.shape)

                count += 1
                has_cost = count % 100 == 0
                if has_cost:
                    cost = compute_cost(AL, Y_t)
                    costs.append(cost)

                    if is_printable_cost:
                        print("Cost after epoch: %i, batch: %i, : %f " %(i+1, t+1, cost))

                dZL = loss_backward(AL, Y_t)

                grads = p.backward_propagation(dZL, caches, layers)

                parameters = commons.update_parameters(L, parameters, grads, self.learning_rate)
        
        return parameters, costs
