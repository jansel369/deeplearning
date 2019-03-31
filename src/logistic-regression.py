import numpy as np

from commons import *

def initialize_params(dim):
    w = np.zeros((dim, 1))
    b = 0

    return w, b

def propagate(w, b, X, Y):
    # w - wight vector (n, 1)
    # b - bias integer
    # X - input matrix (n, m)
    # Y - output label (1, m)
    m = X.shape[1]
    
    # forward progaragtion
    Z = np.dot(w.T, X) + b
    A = sigmoid(z)
    cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # backward propagation
    dz = A - Y
    dw = (1 / m) * np.dot(X, dz.T)
    db = (1 / m) * np.sum(dz)

    grads = { "dw": dw, "db": db }

    return grads, cost

def optimize(w, b, X, Y, iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = { "w": w, "b": b }
    grads = { "dw": dw, "db": db }

    return params, grads, costs

