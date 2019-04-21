import torch as pt

linear = 'linear'
sigmoid = 'sigmoid'
tanh = 'tanh'
relu = 'relu'
softmax = 'softmax'

def sigmoid_forward(z):
    return 1 / (1 + pt.exp(-z))

def relu_forward(Z):
    
    Z[Z < 0] = 0
    
    return Z

def softmax_forward(Z):
    # e = Z.exp()
    # stable version
    e = pt.exp(Z - Z.max())

    return e / e.sum()

def sigmoid_backward(A):
    # A = cache[0]
    # g'(z)
    return A * (1 - A)

def relu_backward(A):
    # g'(z)

    back = (A > 0).clone().detach()

    return pt.tensor(back, dtype=pt.double, device=A.device)
