import copy

# from .commons import *
# from commons import *
# import core.activation as a
# import core.loss as l
# import core.propagation as p
from backend import loss as l
from backend import activation as a
from backend import cost as c
# from core import activation_forward as af
# from core import activation_backward as ab
from backend import gradient as g
from backend import propagation as p
from backend import prediction as pred

# print(l.binary_crossentropy)
# print(cost.binary_crossentropy_cost)

def update_parameters(L, parameters, grads, learning_rate):
    for l in range(1, L):

        W_l = "W" + str(l)
        b_l = "b" + str(l)

        parameters[W_l] = parameters[W_l] - learning_rate * grads["dW" + str(l)]
        parameters[b_l] = parameters[b_l] - learning_rate * grads["db" + str(l)]
    
    return parameters
