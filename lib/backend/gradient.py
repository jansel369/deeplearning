# from . import loss as l

def binary_crossentropy_backward(AL, Y):
    """ calculate the gradient
        summary of derivative dz = da/dz.dL/da with sigmoid
    """
    return AL - Y

def categorical_crossentoropy_backward(AL, Y):
    """ calculate the gradient
        summary of derivative dz = da/dz.dL/da with softmax
    """
    return AL - Y
