import torch as pt

def sigmoid(z):
    return 1 / (1 + pt.exp(-z))

def relu(z):
    return pt.max(z, 0)

def sigmoid_backward(dA, cache):
    A = cache[0]
    # dA * g'(z)
    return dA * A * (1 - A)

def relu_backward(dA, cache)
    A = cache[0]
    # dA * g'(z)
    return dA * pt.tensor((A > 0).clone(), dtype=pt.double, device=dA.device)

def d_cross_entropy_loss(AL, Y):
    return - (pt.devide(Y, AL) - pt.devide(1 - Y, 1 - AL))
