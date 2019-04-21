import torch as pt

categorical_crossentropy = 'categorical_crossentropy'
binary_crossentropy = 'binary_crossentropy'

# def d_cross_entropy_loss(AL, Y):
#     return - (pt.devide(Y, AL) - pt.devide(1 - Y, 1 - AL))

def binary_crossentropy_backward(AL, Y):
    """
        summary of derivative dz = da/dz.dL/da with sigmoid
    """
    return AL - Y

def categorical_crossentoropy_backward(AL, Y):
    """
        summary of derivative dz = da/dz.dL/da with softmax
    """

    return AL - Y

def binary_crossentropy_cost(AL, Y):

    m = Y.shape[1]
    cost =  - (1 / m) * (Y * pt.log(AL) + (1 - Y) * pt.log(1 - AL)).sum()

    return cost

def categorical_crossentropy_cost(AL, Y):
    m = Y.shape[1]
    cost = - (1 / m) * (Y * pt.log(AL)).sum()
