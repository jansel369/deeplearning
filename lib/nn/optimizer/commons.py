from backend import activation as a
from backend import gradient as g


""" constructing backward prop functions
    naming format: (activation + '_')<optimizer>_<type>_<method>
"""

""" parameters update
"""

def weight_std_grad(m):
    def f(dZ, A_prev):
        return (1 / m) * dZ.mm(A_prev.t())
    
    return f

def bias_std_grad(m):
    def f(dZ):
        return (1 / m) * dZ.sum(dim=1, keepdim=True)

    return f


def batch_norm_grad(learning_rate, m):
    """ to calculate batch norm gradients, dZ, dgamma, dbeta
    """
    def gd_bn_update(dZ_tilda, cache, parameters):
        current_cache, next_cache = cache
        gamma, beta, mu, mu_dev, var, gamma_i, Z_norm, epsilon = current_cache

        dgamma = (dZ_tilda * Z_norm).sum(1, True)
        dbeta = dZ_tilda.sum(1, True)

        dZ_norm = dZ_tilda * gamma
        dvar =  ( dZ_norm * mu_dev * (-0.5) * (gamma_i ** (-3)) ).sum(1, True)
        dmu =  (-dZ_norm / gamma_i).sum(1, True) + dvar * (-2 / m) * mu_dev.sum(1, True)
        dZ = dZ_norm / gamma_i + dvar * (2 / m) * mu_dev + dmu / m

        return (dZ, dgamma, dbeta, gamma, beta), next_cache, parameters

def liniar_std_grad(activation_backward):
    """ to calculate standard liniar gradient
    """
    def compute_liniar_std_grad(dZ, cache, parameters):
        current_cache, next_cache = cache
        A, W_preced, b_preced = current_cache
        
        dZ = W_preced.t().mm(dZ) * activation_backward(A)

        return dZ, next_cache, parameters

    return compute_liniar_std_grad


def construct_backwards(update_dict, layers, learnibng_rate, m):
    """ temporary constructor
        todo: batch norm
        backwards - array of backward functions
                  - 2 hidden, 1 output layers
                    - format: [update, liniar_grad, update, liniar_grad, update]
    """

    backwards = []
    # add first update backward
    backwards.append(update_dict['std_update'](learnibng_rate, m))

    for l in reversed(range(1, len(layers) - 1)):
        layer = layers[l]
        activation = layer.activation

        # add next grad calculation
        backwards.append(liniar_std_grad(a.activation_backward_dict[activation]))

        # Add parameter update function
        backwards.append(update_dict['std_update'](learnibng_rate, m))

    return backwards


""" Backward propagation
"""
def backward_propagation(backwards, loss):
    grad_loss = g.loss_backward_dict[loss]
    
    def f(AL, Y, cache):
        parameters = None

        dZ = grad_loss(AL, Y)
        for backward in backwards:
            # print('running backward: ', backward.__name__)

            dZ, cache, parameters = backward(dZ, cache, parameters)

        return parameters

    return f
