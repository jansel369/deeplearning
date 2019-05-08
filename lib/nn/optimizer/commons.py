from backend import activation as a
from backend import gradient as g


""" constructing backward prop functions
    naming format: (activation + '_')<optimizer>_<type>_<method>
"""

""" parameters update
"""

# def weight_std_grad(m):
#     to_avg = 1 / m
    
#     def f(dZ, A_prev):
#         return to_avg * dZ.mm(A_prev.t())
    
#     return f

# def bias_std_grad(m):
#     to_avg = 1 / m
    
#     def f(dZ):
#         return to_avg * dZ.sum(dim=1, keepdim=True)

#     return f


# def batch_norm_grad(learning_rate, m):
#     """ to calculate batch norm gradients, dZ, dgamma, dbeta
#     """
#     to_avg = 1 / m

#     def bn_grad_backward(dZ_tilda, cache, parameters):
#         current_cache, next_cache = cache
#         gamma, beta, mu, mu_dev, var, gamma_i, Z_norm, epsilon = current_cache

#         dgamma = to_avg * (dZ_tilda * Z_norm).sum(1, True)
#         dbeta = to_avg * dZ_tilda.sum(1, True)

#         dZ_norm = dZ_tilda * gamma
#         dvar =  ( dZ_norm * mu_dev * (-0.5) * (gamma_i ** 3) ).sum(1, True)
#         dmu =  (-dZ_norm * gamma_i).sum(1, True) + dvar * (-2 / m) * mu_dev.sum(1, True)
#         dZ = dZ_norm * gamma_i + (2 / m) * dvar * mu_dev + to_avg * dmu
        
#         return (dZ, dgamma, dbeta, gamma, beta), next_cache, parameters

#     return bn_grad_backward

def liniar_std_grad(activation_backward):
    """ to calculate standard liniar gradient
    """
    def compute_liniar_std_grad(dZ, cache, parameters):
        current_cache, next_cache = cache
        A, W_preced, b_preced = current_cache
        
        dZ = W_preced.t().mm(dZ) * activation_backward(A)

        return dZ, next_cache, parameters

    return compute_liniar_std_grad

def param_grad(learning_rate, m):
    weight_grad = weight_std_grad(m)
    bias_grad = bias_std_grad(m)
    def param_grad_backward(dZ, cache, parameters):
        

        dW = weight_grad(dZ, )

    return param_grad_backward
    

# def construct_backwards(update_dict, layers, learnibng_rate, m):
#     """ helper functions that returns array of backward propagation functions
#         backwards - array of backward functions
#                   - 2 hidden, 1 output layers
#                     - format: [update, liniar_grad, (batch_norm_grad,) std/bn: update, liniar_grad, update]
#     """

#     backwards = []
#     # add first update backward
#     backwards.append(update_dict['std_update'](learnibng_rate, m))

#     for l in reversed(range(1, len(layers) - 1)):
#         layer = layers[l]
#         activation = layer.activation
#         update_fn_name = 'std_update'

#         # add next grad calculation
#         # calculates the liniear gradients dZ
#         backwards.append(liniar_std_grad(a.activation_backward_dict[activation]))

#         if layer.batch_norm: # to calculates batch norm gradients
#             update_fn_name = 'bn_update'
#             backwards.append(batch_norm_grad(learnibng_rate, m))

#         # Add parameter update function for batch norm or standard parameters
#         backwards.append(update_dict[update_fn_name](learnibng_rate, m))

#     return backwards


""" Backward propagation
"""
