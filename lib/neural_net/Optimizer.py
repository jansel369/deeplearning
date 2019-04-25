import copy

# from .commons import *
# from commons import *
# import core.activation as a
# import core.loss as l
# import core.propagation as p
from backend import loss as l
from backend import activation as a
from backend import cost as c
# from core import activation_forward as af
# from core import activation_backward as ab
from backend import gradient as g
from backend import propagation as p
from backend import prediction as pred

# print(l.binary_crossentropy)
# print(cost.binary_crossentropy_cost)

loss_backward_dict = {
    l.categorical_crossentropy: g.categorical_crossentoropy_backward,
    l.binary_crossentropy: g.binary_crossentropy_backward,
}

activation_backward_dict = {
    a.sigmoid: a.sigmoid_backward,
    a.relu: a.relu_backward,
}

predict_accuracy_dict = {
    l.binary_crossentropy: pred.binary_crossentropy_predict_accuracy,
    l.categorical_crossentropy: pred.categorical_crossentoropy_predict_accuracy,
}

def predict(X, Y, parameters, layers, loss):

    return predict_accuracy_dict[loss](X, Y, parameters, layers)

def forward_propagation(X, parameters, layers):

    caches = []
    Al = X
    L = len(layers)
    
    for l in range(1, L):

        layer = layers[l]
        A_prev = Al

        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]

        Zl = p.liniar_forward(A_prev, Wl, bl)
        Al = p.activation_forward(Zl, layer['activation'])

        caches.append((A_prev, Wl))

    return Al, caches
    

def backward_propagation(dZL, caches, layers):
    grads = {}
    dZl = dZL
    # L = len(layers)
    m = dZL.shape[1]

    for l in reversed(range(len(caches))):

        # print("l: ", l)
        (A_prev, Wl) = caches[l]

        grads['dW' + str(l + 1)] =  (1 / m) * dZl.mm(A_prev.t())
        grads['db' + str(l + 1)] = (1 / m) * dZl.sum(dim=1, keepdim=True)

        if l > 0:
            prev_activation = layers[l]['activation']
            # print('prev activation: ', l,  prev_activation)
            dZl = Wl.t().mm(dZl) * activation_backward_dict[prev_activation](A_prev)

    return grads

def update_parameters(L, parameters, grads, learning_rate):
    for l in range(1, L):

        W_l = "W" + str(l)
        b_l = "b" + str(l)

        parameters[W_l] = parameters[W_l] - learning_rate * grads["dW" + str(l)]
        parameters[b_l] = parameters[b_l] - learning_rate * grads["db" + str(l)]
    
    return parameters

class GradientDescent():

    def __init__(self, learning_rate, iterations, loss):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = loss
    
    def optimize(self, X, Y, parameters, config, is_printable_cost):
        costs = []
        layers = config['layers']
        L = len(layers)

        compute_cost = c.costs_dict[self.loss]
        loss_backward = loss_backward_dict[self.loss]

        for i in range(self.iterations):

            has_cost = i % 100 == 0

            # print(i)

            AL, caches = forward_propagation(X, parameters, layers)

            if has_cost:
                cost = compute_cost(AL, Y)
                costs.append(cost)

                if is_printable_cost:
                    print("Cost after iteration %i: %f " %(i, cost))

            dZL = loss_backward(AL, Y)

            grads = backward_propagation(dZL, caches, layers)

            parameters = update_parameters(L, parameters, grads, self.learning_rate)

        return parameters, costs

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
        loss_backward = loss_backward_dict[self.loss]

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

                AL, caches = forward_propagation(X_t, parameters, layers)
                # print(AL.shape)

                count += 1
                has_cost = count % 100 == 0
                if has_cost:
                    cost = compute_cost(AL, Y_t)
                    costs.append(cost)

                    if is_printable_cost:
                        print("Cost after epoch: %i, batch: %i, : %f " %(i+1, t+1, cost))

                dZL = loss_backward(AL, Y_t)

                grads = backward_propagation(dZL, caches, layers)

                parameters = update_parameters(L, parameters, grads, self.learning_rate)
        
        return parameters, costs