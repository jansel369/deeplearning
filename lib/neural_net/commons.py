# from commons import *
import torch as pt
from backend import prediction as pred

# @todo: improve initialization

def predict(X, Y, parameters, layers, loss):

    return pred.predict_accuracy_dict[loss](X, Y, parameters, layers)

def init_params(layers, device=pt.device("cpu")):
    params = {}

    for l in range(len(layers) - 1):
        n = layers[l + 1]["units"]
        n_prev = layers[l]["units"]

        params["W" + str(l+1)] = pt.randn(n, n_prev, dtype=pt.double, device=device) * 0.01
        params["b" + str(l+1)] = pt.zeros(n, 1, dtype=pt.double, device=device)
    
    return params

def get_device():
    return pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")

def from_numpy(t, device=get_device(), dtype=pt.double):
    return pt.tensor(t.tolist(), device=device, dtype=dtype)
