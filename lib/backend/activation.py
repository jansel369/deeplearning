import torch as pt

sigmoid = 'sigmoid'
tanh = 'tanh'
relu = 'relu'
softmax = 'softmax'
 

def sigmoid_forward(z):
    return 1 / (1 + pt.exp(-z))

def relu_forward(Z):
    
    Z[Z < 0] = 0
    
    return Z

def softmax_forward(Z, ndim=0):
    # e = Z.exp()
    # stable version
    e = pt.exp(Z - Z.max(ndim)[0])

    return e / e.sum(ndim)

def sigmoid_backward(A):
    # g'(z)
    return A * (1 - A)

def relu_backward(A):
    # g'(z)
    return (A > 0).double()

activations_dict = {
    sigmoid: sigmoid_forward,
    relu: relu_forward,
    softmax: softmax_forward,
}

activation_backward_dict = {
    sigmoid: sigmoid_backward,
    relu: relu_backward,
}
