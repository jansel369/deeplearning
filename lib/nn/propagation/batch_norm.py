import torch as pt
# from backend import gradient as g

""" batch norm
    https://arxiv.org/abs/1502.03167v1
"""

def batch_norm_forward(Z, params, has_cache, cache):
    epsilon=1.001e-5
    current_params, next_params = params
    gamma, beta = current_params
    m = Z.shape[0]
    to_avg = 1 / m

    mu = to_avg * Z.sum(0, True) # mean, sum all the rows of Z
    mu_dev = Z - mu # deviation from mean
    var = to_avg * (mu_dev ** 2).sum(0, True) # variance
    gamma_i = 1 / (var + epsilon).sqrt() # gamma identity
    Z_norm = mu_dev * gamma_i # Z normalized
    Z_tilda = gamma * Z_norm + beta # batch normalized

    cache = (((mu_dev, gamma_i, Z_norm), current_params), cache) if has_cache else None

    return Z_tilda, next_params, cache

def batch_norm_grad_a():
    """ Backprop function that calculates dL/dZ from batch norm backward
    """
    def bn_grad_i(optimizer, to_avg):
        def bn_grad_backward(dZ_tilda, param_grad, cache, parameters):
            current_cache, next_cache = cache
            bn_cache, [gamma, beta] = current_cache
            mu_dev, gamma_i, Z_norm = bn_cache

            dZ_norm = dZ_tilda * gamma
            dvar =  -0.5 * (dZ_norm * mu_dev * (gamma_i ** 3)).sum(0, True)
            dmu =  (dZ_norm * (-gamma_i)).sum(0, True) - 2 * to_avg * dvar * mu_dev.sum(0, True)
            dZ = dZ_norm * gamma_i + 2 * to_avg * dvar * mu_dev + to_avg * dmu
            
            return dZ, param_grad, next_cache, parameters
        
        return bn_grad_backward

    return bn_grad_i

def bn_param_grad_a():
    """ Backprop functions that calculate grad dL/dgamma & dL/dbeta
    """
    def param_grad(optimizer, to_avg):
        def bn_grad(dZ_tilda, param_grad, cache, parameters):
            current_cache, _ = cache
            (_, _, Z_norm), _ = current_cache

            dgamma = to_avg * (dZ_tilda * Z_norm).sum(0, True)
            dbeta = to_avg * dZ_tilda.sum(0, True)

            return dZ_tilda, [dgamma, dbeta], cache, parameters
        
        return bn_grad
    return param_grad