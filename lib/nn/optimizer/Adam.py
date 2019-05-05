import torch as pt


from nn.CostEvaluator import CostEvaluator
from nn import propagation as p
from .StochasticGradientDeschent import StochasticGradientDescent

from . import commons as c
from .Momentum import vel_bias_f, vel_weight_f
from .RMSProp import prop_bias_f, prop_weight_f

def std_update(beta1, beta2, epsilon):
    vel_weight = vel_weight_f(beta1)
    vel_bias = vel_bias_f(beta1)
    prop_weight = prop_bias_f(beta2, epsilon)
    prop_bias = prop_bias_f(beta2, epsilon)

    def adam_update(learning_rate, m):
        """ adam standard update
        """
        weight_grad = c.weight_std_grad(m)
        bias_grad = c.bias_std_grad(m)
        
        VdW = 0
        Vdb = 0
        SdW = 0
        Sdb = 0
        t = 1 # iteration

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

            VdW = vel_weight(VdW, dW)
            Vdb = vel_bias(Vdb, db)

            vcn = 1 - (beta1 ** t)
            VdW_c = VdW / vcn # _c: corrected
            Vdb_c = Vdb / vcn

            SdW = prop_weight(SdW, dW)
            Sdb = prop_bias(Sdb, db)

            scn = 1 - (beta2 ** t)
            SdW_c = SdW / scn
            Sdb_c = Sdb / scn

            W -= learning_rate * ( VdW_c / (SdW_c + epsilon).sqrt() )
            b -= learning_rate * ( Vdb_c / (Sdb_c + epsilon).sqrt() )

            t += 1 # update t by 1 per iteration

            return dZ, cache, ((W, b), parameters)

        return update

    return adam_update

class Adam(StochasticGradientDescent):
    def __init__(self, loss, epochs, batch_size=64, learning_rate=3e-4, beta1=0.9, beta2=0.999, epsilon=10e-8):
        super().__init__(loss, epochs, batch_size, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def optimize(self, X, Y, parameters, config, is_printable_cost):
        update_dict = {
            "std_update": std_update(self.beta1, self.beta2, self.epsilon)
        }

        return super().optimize(X, Y, parameters, config, is_printable_cost, update_dict)
