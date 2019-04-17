import torch as pt
# from .commons import *
import copy
# import optimizer

def create_layer_config(size):
    layer = {
        'size': size,
        'activation': ''
    }

    return layer

def Input(size):
    layer = create_layer_config(size)

    config = {
        'layers': [layer]
    }

    return config

def Layer(size):
    def a(config):
        config = copy.deepcopy(config)

        layer = create_layer_config(size)

        config["layers"].append(layer)

        return config

    return a
