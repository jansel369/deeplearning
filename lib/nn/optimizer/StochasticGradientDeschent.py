# from backend import loss as l
from backend import activation as a
# from backend import cost as c
# from backend import gradient as g
from nn import propagation as p
# from backend import prediction as pred

from . import commons as c
from .GradientDescent import update_dict

from nn.CostEvaluator import CostEvaluator

class StochasticGradientDescent:
    def __init__(self, learning_rate, epochs, batch_size, loss):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
    
    def optimize(self, X, Y, parameters, config, is_printable_cost):
        layers = config.layers
        m = Y.shape[1]
        batch_count = int(m / self.batch_size)

        cost_evaluator = CostEvaluator(self.loss, self.learning_rate, print_cost=is_printable_cost)

        forwards = config.forwards
        backwards = c.construct_backwards(update_dict, layers, self.learning_rate, self.batch_size)

        forward_prop = p.forward_propagation(forwards, has_cache=True) # set has_cache=True for backprap training
        backward_prop = c.backward_propagation(backwards, self.loss)

        X_batches = X.split(self.batch_size, 1)
        Y_batches = Y.split(self.batch_size, 1)

        for i in range(self.epochs):
            for t in range(batch_count):
                X_b = X_batches[t]
                Y_b = Y_batches[t]

                AL, cache = forward_prop(X, parameters)

                parameters = backward_prop(AL, Y, cache)
                
                cost_evaluator.stochastic_cost(i, t, AL, Y)

        return parameters, cost_evaluator