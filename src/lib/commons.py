import torch as pt

def sigmoid(z):
    return 1 / (1 + pt.exp(-z))
