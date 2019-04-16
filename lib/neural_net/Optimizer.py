import copy

# from .commons import *
from commons import *

activations_dict = {
    'sigmoid': sigmoid,
    'relu': relu,
    'soft_max': soft_max,
}

costs_dict = {
    'cross_entropy': compute_cross_entropy_cost,
}

loss_backward_dict = {
    'sigmoid_cross_entropy': sigmoid_cross_entropy_backward,
    'softmax_cross_entropy': softmax_cross_entropy_backward,
}

activation_backward_dict = {
    'sigmoid': sigmoid_backward,
    'relu': relu_backward
}

def liniar_forward(A_prev, W, b):

    Z = W.mm(A_prev) + b

    return Z

def activation_forward(A, layer):
    activation = layer['activation']

    return activations_dict[activation] if activation != '' else A

def forward_propagation(X, parameters, layers):

    caches = []
    Al = X
    L = len(layers)

    for l in range(1, L + 1):
        layer = layers[l]
        A_prev = Al

        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]

        Zl = liniar_forward(A_prev, Wl, bl)
        Al = activation_forward(Al, layer)

        caches.append((A_prev, Wl))

    return Al, caches
    

def backward_propagation(dzL, Y, caches, layers):
    grads = {}
    dzl = dzL
    L = len(layers)
    m = Y.shape[1]

    for l in reversed(range(L)):
        (A_prev, Wl) = caches[l]
        activation = layers[l]['activation']

        # compute grads
        grads['dW' + str(l + 1)] = (1 / m) * dzl.mm(A_prev.t())
        grads['db' + str(l + 1)] = (1 / m) * dzl.sum(dim=1, keepdim=True)

        # compute next dz
        # dz_prev = dA * g' = dz x A_prev * g'(z)
        dzl = Wl.t().mm(dzl) * activation_backward_dict[activation]
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters)

    for l in range(L):
        W_l = "W" + str(l+1)
        parameters[W_l] = parameters[W_l] - learning_rate * grads["dW" + str(l + 1)]
        b_l = "b" + str(l+1)
        parameters[b_l] = parameters[b_l] - learning_rate * grads["db" + str(l + 1)]
    
    return parameters

def Adam(learning_reate, batch_size, epoch):
    def f(config): 
        config = copy.deepcopy(config)
        
        # @todo: finish
        config['optimization'] = {
            
        }

        return config

    return f

def GradientDescent(learning_rate, iterations, loss="cross_entropy"):
    def f(config):
        config = copy.deepcopy(config)

        config['optimization'] = {
            'learning_rate': learning_rate,
            'iterations': iterations,
            'loss': loss,
        }

        return config

    return f


def gradient_descent_optimization(X, Y, parameters, config, is_printable_cost=False):
    costs = []
    optimization = config['optimization']
    iterations = optimization['iterations']
    loss = optimization['loss']
    layers = config['layers']
    learning_rate = optimization['learning_rate']

    compute_cost = costs_dict[loss]
    loss_backward = loss_backward_dict[layers[-1]['activation'] + loss]

    for i in range(iterations):

        has_cost = i % 100 == 0

        AL, caches = forward_propagation(X, parameters, layers)
        if has_cost:
            cost = compute_cost(AL, Y)
            costs.append(cost)

            if is_printable_cost:
                print("Cost after iteration %i: %f percent" %(i, cost * 100))

        dZL = loss_backward(AL, Y)

        grads = backward_propagation(dZL, Y, caches, layers)

        parameters = update_parameters(parameters, grads, learning_rate)

    return parameters, costs
