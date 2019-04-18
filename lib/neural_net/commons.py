from commons import *
import torch as pt

# @todo: improve initialization
def init_params(layers, device=pt.device("cpu")):
    params = {}

    for l in range(len(layers) - 1):
        n = layers[l + 1]["size"]
        n_prev = layers[l]["size"]

        params["W" + str(l + 1)] = pt.randn(n, n_prev, dtype=pt.double, device=device) * 0.01
        params["b" + str(l + 1)] = pt.zeros(n, 1, dtype=pt.double, device=device)
    
    return params


def test_commons():
    print("test from commons: ", __name__)

### OLD JUNK


# def liniar_backward(dz, cache):
#     A_prev, W, b = cache
#     m = A_prev.shape[1]

#     dW = (1 / m) * dz.mm(A_prev.t())
#     db = (1 / m) * dz.sum(dim=1, keepdim=True)

#     dA_prev = W.t().mm(dz)

#     return dA_prev, dW, db

# def liniar_activation_backward_old(dA, cache, activation):

#     if activation == "sigmoid":
#         dZ = sigmoid_backward()
    
#     if activation == "relu":
#         dZ = relu_backward(dA, cache)

#     dA_prev, dW, db = linear_backward(dZ, cache)

#     return dA_prev, dW, db

# def backward_propagation_old(AL, Y, caches):
#     grads = {}
#     L = len(caches)
#     m = AL.shape[1]

#     dAL = d_cross_entropy_loss(AL, Y)

#     # current_cache = caches[L - 1]
#     # grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
#     grads["dA" + str(L)] = dAL

#     for l in reversed(range(L)):
#         current_cache = caches[l]
#         # activation = current_cache[4]

#         dA_prev_temp, dW_temp, db_temp = liniar_activation_backward(grads["dA" + str(l + 1)], current_cache, activation)

#         grads["dA" + str(l)] = dA_prev_temp
#         grads["dW" + str(l + 1)] = dW_temp
#         grads["db" + str(l + 1)] = db_temp

#     return grads

# def activation_backward(dz, cache, layer):

def get_device():
    return pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

def from_numpy(t, device=get_device(), dtype=pt.double):
    return pt.tensor(t.tolist(), device=device, dtype=dtype)
