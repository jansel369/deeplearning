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

