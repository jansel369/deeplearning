import torch as pt
# def d_cross_entropy_loss(AL, Y):
#     return - (pt.devide(Y, AL) - pt.devide(1 - Y, 1 - AL))

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

def binary_crossentropy_cost(AL, Y):
    m = Y.shape[1]
    cost =  - (1 / m) * (Y * pt.log(AL) + (1 - Y) * pt.log(1 - AL)).sum()

    return cost

def categorical_crossentropy_cost(AL, Y):
    m = Y.shape[1]
    cost = - (1 / m) * (Y * pt.log(AL)).sum()

    return cost

""" ACTIVATION
"""

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
    return (A > 0).double()

    # return pt.tensor(g, dtype=pt.double, device=A.device)

""" PROPAGATION
"""

activations_dict = {
    a.sigmoid: a.sigmoid_forward,
    a.relu: a.relu_forward,
    a.softmax: a.softmax_forward,
}


def liniar_forward(A_prev, W, b):
    Z = W.mm(A_prev) + b

    return Z

def activation_forward(Z, activation):
    A = activations_dict[activation](Z)

    return A


""" PREDICTION
"""


def predict(X, parameters, layers):
    Al = X
    L = len(layers)


    for l in range(1, L):
        layer = layers[l]
        A_prev = Al

        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]

        Zl = p.liniar_forward(A_prev, Wl, bl)
        Al = p.activation_forward(Zl, layer['activation'])
    
    return Al

def categorical_crossentoropy_predict_accuracy(X, Y, parameters, layers):
    AL = predict(X, parameters, layers)

    equality = Al.argmax(1).eq(Y.argmax(1))

    return equality.double().mean() * 100

def binary_crossentropy_predict_accuracy(X, Y, parameters, layers):
    AL = predict(X, parameters, layers)

    return 100 - (AL - Y).abs().double().mean() * 100