"""
reference: https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html

"""
from collections import namedtuple

import torch as pt


leaky_value = 0.01 

def sigmoid_forward(Z):
    return 1 / (1 + pt.exp(-Z))

def relu_forward(Z):
    
    Z[Z < 0] = 0
    
    return Z

def leaky_relu_forward(Z):
    Z[Z < 0] = Z[Z < 0] * leaky_value

    return Z

def tanh_forward(Z):
    e_p = pt.exp(Z)
    e_n = pt.exp(-Z)
    
    G = (e_p - e_n) / (e_p + e_n)

    return G

def softmax_forward(Z, ndim=0):
    # e = Z.exp()
    # stable version
    e = pt.exp(Z - Z.max(ndim)[0])

    return e / e.sum(ndim)

def sigmoid_backward(A):
    # g'(z)
    g = A * (1 - A)
    return g

def relu_backward(A):
    # g'(z)
    g = (A > 0).double()
    return g

def leaky_relu_backward(A):
    G = (A > 0).double()

    G[G < 0] = leaky_value

    return G

def tanh_backward(A):
    return 1 - A.sqrt()

def softmax_backward(A):
    return 0


Activation = namedtuple('Activation', 'name, forward, backward')

sigmoid = Activation('sigmoid', sigmoid_forward, sigmoid_backward)
tanh = Activation('tanh', tanh_forward, tanh_backward)
relu = Activation('relu', relu_forward, relu_backward)
softmax = Activation('softmax', softmax_forward, softmax_backward)
leaky_relu = Activation('leaky_relu', leaky_relu_forward, leaky_relu_backward)
