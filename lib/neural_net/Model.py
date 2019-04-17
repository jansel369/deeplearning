import copy
import torch as pt
from .commons import *
from .Optimizer import *
import matplotlib.pyplot as plt

# import utils.plot_cost as plot

# def compose_forward_propagation(layers):
#     forward

#     for l in range(len(layers)):
#         layer = layers[l + 1]


#     return forward

optimizer_dict = {
    'gradient_descent': gradient_descent_optimization,
}

def plot_cost(costs, learning_rate):
    plt.plot(costs)
    plt.ylabel("costs")
    plt.xlabel("iterations / 100s")
    plt.title("Logistic Regression (a=" + str(learning_rate) + ")")
    plt.show()

class Model():
    def __init__(self, config):
        self._config = copy.deepcopy(config)

        # layers = config['layers']
        
    
    # def optimization(self, optimizer):
    #     self.optimizer = optimizer

    def fit(self, X_train, Y_train, is_printable_cost=False):
        device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")
        optimizer = self._config['optimization']['optimizer']

        parameters = init_params(self._config["layers"], device)

        # return self.optimizer(X_train, Y_train, parameters)

        parameters, costs = optimizer_dict[optimizer](X_train, Y_train, parameters, self._config, is_printable_cost)
        
        self._parameters = parameters;

        plot_cost(costs, self._config['optimization']['learning_rate'])
    
    def evaluate(self, X, Y):
        AL = predict(X, Y, self._config['layers'])

        # return parameters, costs
# def MakeModel(config):
    