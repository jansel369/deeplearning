import torch as pt

from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent
from . import commons as c
from .Momentum import vel_grad_f
from .RMSProp import prop_grad_f

def adam_param_update_f(epsilon):
    def update_param(learning_rate, param, vdc, sdc):
        param -= learning_rate * (vdc / (sdc + epsilon).sqrt())

        return param

    return update_param

def corrected_f(beta, t):
    cd = 1 - (beta ** t)

    def corrected(op_grad):
        return op_grad / cd

    return corrected

def std_update(beta1, beta2, epsilon):
    vel_grad = vel_grad_f(beta1)
    rms_grad = prop_grad_f(beta2)
    param_update = adam_param_update_f(epsilon)

    def adam_update(learning_rate, m):
        """ adam standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        VdW = 0
        Vdb = 0
        SdW = 0
        Sdb = 0
        t = 1 # iteration, should start at 1 to avoid deviding by 0

        def update(dZ, cache, parameters):
            """Adam standard update
                cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
                parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
            """
            nonlocal SdW, Sdb, VdW, Vdb, t

            current_cache, next_cache = cache
            A_prev, W, b = current_cache

            dW = weight_grad(dZ, A_prev)
            db = bias_grad(dZ)

            VdW = vel_grad(VdW, dW)
            Vdb = vel_grad(Vdb, db)

            vel_corrected = corrected_f(beta1, t)
            VdW_c = vel_corrected(VdW)
            Vdb_c = vel_corrected(Vdb)

            SdW = rms_grad(SdW, dW)
            Sdb = rms_grad(Sdb, db)

            rms_corrected = corrected_f(beta2, t)
            SdW_c = rms_corrected(SdW)
            Sdb_c = rms_corrected(Sdb)

            W = param_update(learning_rate, W, VdW_c, SdW_c)
            b = param_update(learning_rate, b, Vdb_c, Sdb_c)

            t += 1 # update t by 1 per iteration

            return dZ, cache, ((W, b), parameters)

        return update

    return adam_update

def adam_batch_norm_update_f(beta1, beta2, epsilon):
    vel_grad = vel_grad_f(beta1)
    rms_grad = prop_grad_f(beta2)
    param_update = adam_param_update_f(epsilon)

    def adam_bn_update(learning_rate, m):
        """ adam standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        VdW = 0
        Vdgamma = 0
        Vdbeta = 0
        SdW = 0
        Sdgamma = 0
        Sdbeta = 0
        t = 1 # iteration, should start at 1 to avoid deviding by 0

        def update(cache1, cache2, parameters):
            nonlocal VdW, Vdgamma, Vdbeta, SdW, Sdgamma, Sdbeta, t

            dZ, dgamma, dbeta, gamma, beta = cache1
            current_cache, next_cache = cache2
            A_prev, W, b = current_cache

            dW = weight_grad(dZ, A_prev)

            VdW = vel_grad(VdW, dW)
            Vdgamma = vel_grad(Vdgamma, dgamma)
            Vdbeta = vel_grad(Vdbeta, dbeta)

            vel_corrected = corrected_f(beta1, t)
            VdW_c = vel_corrected(VdW)
            Vdgamma_c = vel_corrected(Vdgamma)
            Vdbeta_c = vel_corrected(Vdbeta)

            SdW = rms_grad(SdW, dW)
            Sdgamma = rms_grad(Sdgamma, dgamma)
            Sdbeta = rms_grad(Sdbeta, dbeta)

            rms_corrected = corrected_f(beta2, t)
            SdW_c = rms_corrected(SdW)
            Sdgamma_c = rms_corrected(Sdgamma)
            Sdbeta_c = rms_corrected(Sdbeta)

            W = param_update(learning_rate, W, VdW_c, SdW_c)
            gamma = param_update(learning_rate, gamma, Vdgamma_c, Sdgamma_c)
            beta = param_update(learning_rate, beta, Vdbeta_c, Sdbeta_c)

            t += 1 # update t by 1 per iteration

            parameters = ((W, b), ((gamma, beta), parameters))

            return dZ, cache2, parameters

        return update

    return adam_bn_update


class Adam(StochasticGradientDescent):
    def __init__(self, loss, epochs, batch_size=64, learning_rate=3e-4, beta1=0.9, beta2=0.999, epsilon=10e-8):
        super().__init__(loss, epochs, batch_size, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        update_dict = {
            'std_update': std_update(self.beta1, self.beta2, self.epsilon),
            'bn_update': adam_batch_norm_update_f(self.beta1, self.beta2, self.epsilon),
        }

        return super().optimize(X, Y, parameters, config, is_printable_cost, update_dict)
