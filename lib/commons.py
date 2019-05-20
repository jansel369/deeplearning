# from commons import *
import torch as pt

def get_device():
    return pt.device("cuda:0") if pt.cuda.is_available() else pt.device("cpu")

def from_numpy(t, device=get_device(), dtype=pt.double):
    return pt.tensor(t.tolist(), device=device, dtype=dtype)
