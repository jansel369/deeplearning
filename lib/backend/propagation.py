from .activation import activations_dict

def liniar_forward(A_prev, W, b):
    Z = W.mm(A_prev) + b

    return Z

def activation_forward(Z, activation):
    A = activations_dict[activation](Z)

    return A

