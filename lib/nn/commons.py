# from commons import *
import torch as pt
from backend import prediction as pred
from backend import initialization as init


def predict(X, Y, parameters, layers, loss):

    return pred.predict_accuracy_dict[loss](X, Y, parameters, layers)


def get_device():
    return pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")

def from_numpy(t, device=get_device(), dtype=pt.double):
    return pt.tensor(t.tolist(), device=device, dtype=dtype)
