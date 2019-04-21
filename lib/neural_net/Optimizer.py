import copy

# from .commons import *
# from commons import *
import activation as a
import loss as l

activations_dict = {
    'sigmoid': a.sigmoid_forward,
    'relu': a.relu_forward,
    'softmax': a.softmax_forward,
}

costs_dict = {
    'binary_crossentropy': l.binary_crossentropy_cost,
    'categorical_crossentropy': l.categorical_crossentropy_cost,
}

loss_backward_dict = {

    """ Computes layer L gradients
    """

    'binary_crossentropy': l.binary_crossentropy_backward,
    'categorical_crossentoropy': l.categorical_crossentoropy_backward,
}

activation_backward_dict = {
    'sigmoid': a.sigmoid_backward,
    'relu': a.relu_backward
}

def predict(X, Y, parameters, layers):
    Al = X
    L = len(layers)

    for l in range(1, L + 1):
        layer = layers[l]
        A_prev = Al

        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]

        Zl = liniar_forward(A_prev, Wl, bl)
        Al = activation_forward(Al, layer)

    equality = Al.argmax(1).eq(Y.argmax(1))

    return equality.double().mean() * 100


def liniar_forward(A_prev, W, b):

    # print("A prev %s" % A_prev)
    # print("W: %s" % W )
    # print("b: %s" % b)

    Z = W.mm(A_prev) + b

    return Z

def activation_forward(Z, layer):
    activation = layer['activation']

    A = activations_dict[activation](Z)

    return A

def forward_propagation(X, parameters, layers):

    caches = []
    Al = X
    L = len(layers)
    
    for l in range(1, L):

        layer = layers[l]
        A_prev = Al

        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]

        Zl = liniar_forward(A_prev, Wl, bl)

        Al = activation_forward(Zl, layer)

        caches.append((A_prev, Wl))

    return Al, caches
    

def backward_propagation(dzL, Y, caches, layers):
    grads = {}
    dzl = dzL
    L = len(layers)
    m = Y.shape[1]

    for l in reversed(range(L-1)):
        
        # print("l: ", l)

        (A_prev, Wl) = caches[l]

        # print("APrev: %d" % (l), A_prev.shape)

        # compute grads
        grads['dW' + str(l + 1)] = (1 / m) * dzl.mm(A_prev.t())
        grads['db' + str(l + 1)] = (1 / m) * dzl.sum(dim=1, keepdim=True)

        # compute next dz
        # dz_prev = dA * g' = dz x A_prev * g'(z)

        #compute next dz


        # print("the activation: ", activation)

        if l > 0:
            activation = layers[l]['activation']
            # print('activation: ', activation)
            dzl = Wl.t().mm(dzl) * activation_backward_dict[activation](A_prev)

    # print("grads: dwl", grads["dW1"].shape)
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = int(len(parameters) / 2)

    # print("value of L: ", L)

    # print(grads)

    for l in range(L):

        W_l = "W" + str(l+1)
        
        # print(parameters[W_l].shape)
        # print(grads["dW" + str(l + 1)].shape)

        parameters[W_l] = parameters[W_l] - learning_rate * grads["dW" + str(l + 1)]
        b_l = "b" + str(l+1)
        parameters[b_l] = parameters[b_l] - learning_rate * grads["db" + str(l + 1)]
    
    return parameters

class GradientDescent():

    def __init__(self, learning_rate, iterations, loss):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = loss
    
    def optimize(self, X, Y, parameters, config, is_printable_cost):
        costs = []
        layers = config['layers']

        compute_cost = costs_dict[self.loss]
        loss_backward = loss_backward_dict[self.loss]

        for i in range(self.iterations):

            has_cost = i % 50 == 0

            # print(i)

            AL, caches = forward_propagation(X, parameters, layers)

            if has_cost:
                cost = compute_cost(AL, Y)
                costs.append(cost)

                if is_printable_cost:
                    print("Cost after iteration %i: %f percent" %(i, cost * 100))

            dZL = loss_backward(AL, Y)

            grads = backward_propagation(dZL, Y, caches, layers)

            parameters = update_parameters(parameters, grads, self.learning_rate)

        return parameters, costs
