import torch as pt

def sigmoid(z):
    return 1 / (1 + pt.exp(-z))

def relu(Z):
    
    Z[Z < 0] = 0
    
    return Z

def sigmoid_backward(A):
    # A = cache[0]
    # g'(z)
    return A * (1 - A)

def relu_backward(A):
    # g'(z)

    back = (A > 0).clone().detach()

    return pt.tensor(back, dtype=pt.double, device=A.device)

def d_cross_entropy_loss(AL, Y):
    return - (pt.devide(Y, AL) - pt.devide(1 - Y, 1 - AL))

def softmax(Z):
    # e = Z.exp()
    # stable version
    e = pt.exp(Z - Z.max())

    return e / e.sum()

def sigmoid_cross_entropy_backward(AL, Y):
    """
        summary of derivative dz = da/dz.dL/da with sigmoid
    """
    return AL - Y

def softmax_cross_entropy_backward(AL, Y):
    """
        summary of derivative dz = da/dz.dL/da with softmax
    """

    return AL - Y

def compute_cross_entropy_cost(AL, Y):

    m = Y.shape[1]
    cost =  - (1 / m) * (Y * pt.log(AL) + (1 - Y) * pt.log(1 - AL)).sum()

    return cost

def softmax_cost(AL, Y):
    m = Y.shape[1]
    
    cost = - (1 / m) * (Y * AL.log()).sum()

    return cost