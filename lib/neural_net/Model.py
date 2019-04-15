import copy
import .commons import *

def compose_forward_propagation(layers):
    forward

    for l in range(len(layers)):
        layer = layers[l + 1]

        


    return forward




class Model():
    def __init__(self, config):
        self._config = copy.deepcopy(config)

        layers = config['layers']


        
    
    # def optimization(self, optimizer):
    #     self.optimizer = optimizer

    def fit(self, X_train, Y_train):
        device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
        
        parameters = init_params(self.config["layers"], device)

        return self.optimizer(X_train, Y_train, parameters)
        
# def MakeModel(config):
    