import torch as pt

from commons import *


def liniar_forward(W, A, b):
    Z = W.mm(A) + b

    return Z, (A, W, b)

def liniar_activation_forward(A_prev, W, b, activation):
    
    Z, liniar_cache = liniar_forward(W, A_prev, b)

    if activation = "sigmoid":
        A = sigmoid(z)

    if activation = "relu":
        A = relu(z)
    
    cache = liniar_cache + (A, activation)

    return A, cache


def forward_propagation(X, parameters):

    caches = []
    A = X
    L = len(parameters)

    for l in range(1, L + 1):
        A_prev = A
        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]
        activation = "relu" if l < L else "sigmoid"
        A, cache = liniar_activation_forward(A_prev, Wl, bl, activation)
        caches.append(cache)


    return A, caches

def compute_cost(AL, Y):

    m = Y.shape[1]
    cost =  - (1 / m) * (Y * pt.log(AL) + (1 - Y) * pt.log(1 - AL)).sum()

    return cost

def liniar_backward(dz, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * dz.mm(A_prev.t())
    db = (1 / m) * dz.sum(dim=1, keepdim=True)

    dA_prev = W.t().mm(dz)

    return dA_prev, dW, db

def liniar_activation_backward(dA, cache, activation):

    # linear_cache, activation_cache = cache


    if activation = 'sigmoid':
        dZ = sigmoid_backward(dA, cache)
    
    if activation = 'relu'
        dZ = relu_backward(dA, cache)

    dA_prev, dW, db = linear_backward(dZ, cache)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    dAL = d_cross_entropy_loss(AL, Y)

    # current_cache = caches[L - 1]
    # grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L)] = dAL

    for l in reversed(range(L)):
        current_cache = caches[l]
        activation = current_cache[4]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation)

        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters)

    for l in range(L):
        W_l = "W" + str(l+1)
        parameters[W_l] = parameters[W_l] - learning_rate * grads["dW" + str(l + 1)]
        b_l = "b" + str(l+1)
        parameters[b_l] = parameters[b_l] - learning_rate * grads["db" + str(l + 1)]
    
    return parameters
    