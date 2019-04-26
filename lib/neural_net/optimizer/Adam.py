import torch as pt

from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
from backend import propagation as p
from backend import prediction as pred

from .Momentum import init_velocity, update_velocity
from .RMSProp import init_rms, update_rms

def velocity_corrected(velocity, beta1):
    corrected = {}
    
    for key, value in velocity.items():
        corrected[key] = value / (1 - beta1)
    
    return corrected

def rms_corrected(rms, beta2):
    corrected = {}
    
    for key, value in rms.items():
        corrected[key] = value / (1 - beta2)
    
    return corrected

def update_parameters(L, parameters, vel_c, rms_c, learning_rate, epsilon):
    for l in range(1, L):
        l_s = str(l)

        parameters["W"+l_s] -= ( learning_rate * vel_c["VdW"+l_s] / (rms_c["SdW"+l_s] + epsilon).sqrt() )
        parameters["b"+l_s] -= ( learning_rate * vel_c["Vdb"+l_s] / (rms_c["Sdb"+l_s] + epsilon).sqrt() ) 
    
    return parameters

class Adam:
    def __init__(self, learning_rate, iterations, batch_size, loss, beta1=0.9, beta2=0.999, epsilon=10e-8):
        self.learning_rate = learning_rate
        self.epochs = iterations
        self.batch_size = batch_size
        self.loss = loss
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        costs = []
        layers = config['layers']
        L = len(layers)
        m = Y.shape[1]

        compute_cost = c.costs_dict[self.loss]
        loss_backward = g.loss_backward_dict[self.loss]

        batch_iterations = int(m / self.batch_size)
        count = 0

        velocity = init_velocity(parameters)
        rms = init_rms(parameters)

        for i in range(self.epochs):

            for t in range(batch_iterations):
                batch_start = t * self.batch_size
                batch_end = batch_start + self.batch_size

                X_t = X[:, batch_start:batch_end]
                Y_t = Y[:, batch_start:batch_end]

                AL, caches = p.forward_propagation(X_t, parameters, layers)

                count += 1
                has_cost = count % 100 == 0
                if has_cost:
                    cost = compute_cost(AL, Y_t)
                    costs.append(cost)

                    if is_printable_cost:
                        print("Cost after epoch %i, batch %i, : %f " %(i+1, t+1, cost))

                dZL = loss_backward(AL, Y_t)

                grads = p.backward_propagation(dZL, caches, layers)
                
                velocity = update_velocity(grads, velocity, self.beta1)
                rms = update_rms(grads, rms, self.beta2)

                vel_c = velocity_corrected(velocity, self.beta1)
                rms_c = rms_corrected(rms, self.beta2)

                parameters = update_parameters(L, parameters, vel_c, rms_c, self.learning_rate, self.epsilon)
        
        return parameters, costs
