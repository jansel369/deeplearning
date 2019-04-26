import torch as pt

from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
from backend import propagation as p
from backend import prediction as pred

def init_prop(parameters):
    prop = {}

    for param, value in parameters.items():
        prop["Sd"+param] = pt.zeros(value.shape, dtype=pt.double, device=value.device)

    return prop

def update_prop(grads, prop, B):
    for grad_k, grad_v in grads.items():
        vel = "S" + grad_k
        prop[vel] = B * prop[vel] + (1 - B) * (grad_v ** 2)

    return prop

def update_parameters(L, parameters, grads, prop, learning_rate, epsilon):
    for l in range(1, L):
        l_s = str(l)

        parameters["W"+l_s] -= ( learning_rate *  grads["dW"+l_s] / (prop["SdW"+l_s] + epsilon).sqrt() )
        parameters["b"+l_s] -= ( learning_rate * grads["db"+l_s] / (prop["Sdb"+l_s] + epsilon).sqrt() ) 
    
    return parameters

class RMSProp:
    def __init__(self, learning_rate, iterations, batch_size, loss, beta2=0.9, epsilon=10**-8):
        self.learning_rate = learning_rate
        self.epochs = iterations
        self.batch_size = batch_size
        self.loss = loss
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        costs = []
        layers = config['layers']
        prop = init_prop(parameters)
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
                prop = update_prop(grads, prop, self.beta2)
                parameters = update_parameters(L, parameters, grads, prop, self.learning_rate, self.epsilon)
        
        return parameters, costs