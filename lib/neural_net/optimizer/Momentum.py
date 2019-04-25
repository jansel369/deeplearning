import torch as pt

from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
from backend import propagation as p
from backend import prediction as pred

def init_velocity(parameters):
    velocity = {}

    for param, value in parameters.items():
        velocity[param] = pt.zeros(pt.shape, dtype=pt.double, device=value.device)

    return velocity

def update_velocity(grads, velocity, B):
    for grad_k, grad_v in grads.items():
        vel = "V" + grad_k
        velocity[vel] = B * velocity[vel] + (1 - B) * grad_v

    return velocity

def update_parameters(L, parameters, velocity, learning_rate):
    for l in range(1, L):

        W_l = "W" + str(l)
        b_l = "b" + str(l)

        parameters[W_l] = parameters[W_l] - learning_rate * velocity["VdW" + str(l)]
        parameters[b_l] = parameters[b_l] - learning_rate * velocity["Vdb" + str(l)]
    
    return parameters

class Momentum:
    def __init__(self, learning_rate, iterations, batch_size, loss, beta=0.9):
        self.learning_rate = learning_rate
        self.epochs = iterations
        self.batch_size = batch_size
        self.loss = loss
        self.beta = beta

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        costs = []
        layers = config['layers']
        velocity = init_velocity(parameters)
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

                AL, caches = p.forward_propagation(X_t, parameters, layers)

                count += 1
                has_cost = count % 100 == 0
                if has_cost:
                    cost = compute_cost(AL, Y_t)
                    costs.append(cost)

                    if is_printable_cost:
                        print("Cost after epoch %i, batch %i, : %f " %(i+1, t+1, cost))

                dZL = loss_backward(AL, Y_t)

                grads = p.backward_propagation(dZL, caches, layers)
                velocity = update_velocity(grads, velocity, self.beta)
                parameters = update_parameters(L, parameters, velocity, self.learning_rate)
        
        return parameters, costs