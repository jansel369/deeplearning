import torch as pt
# from . import loss as l

def binary_crossentropy_cost(AL, Y):
    m = Y.shape[1]
    cost =  - (1 / m) * (Y * pt.log(AL) + (1 - Y) * pt.log(1 - AL)).sum()

    return cost

def categorical_crossentropy_cost(AL, Y):
    m = Y.shape[1]
    cost = - (1 / m) * (Y * pt.log(AL)).sum()

    return cost
