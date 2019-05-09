
def liniar_forward(A_prev, params, has_cache, cache):
    """cache format
        ((A_prev, [W, b]), ((A_prev, [W, b]),...))
    """

    current_params, next_params = params
    W, b = current_params

    Z = W.mm(A_prev) + b

    cache = ((A_prev, current_params), cache) if has_cache else None

    return Z, next_params, cache

def update_param_a():
    """ Backprop function initialized by specified optimizer update
    """
    def construct_update(optimizer, to_avg):
        return optimizer.param_update_f(optimizer, to_avg)

    return construct_update

def liniar_grad_f(activation_backward):
    """ Backprop function to calculate grad dL/dz
    """
    def liniar_grad_f2(optimizer, to_avg):
        def liniar_grad(dA, param_grad, cache, parameters):
            current_cache, next_cache = cache
            A, preced_param = current_cache

            dZ = dA * activation_backward(A)
            
            return dZ, param_grad, next_cache, parameters
        return liniar_grad
    return liniar_grad_f2



""" Helper functions: Calculating gradient parameters dW, db for liniar and bn
"""

def weight_grad(dZ, avg, A_prev):
    return avg * dZ.mm(A_prev.t())

def bias_grad(dZ, avg):
    return avg * dZ.sum(dim=1, keepdim=True)

def std_params_grad_f(dZ, A_prev, to_avg):
    dW = weight_grad(dZ, to_avg, A_prev)
    db = bias_grad(dZ, to_avg)

    return [dW, db]
    
def bn_prams_grad_f(dZ, A_prev, to_avg):
    dW = weight_grad(dZ, to_avg, A_prev)

    return [dW]

def param_grad_a(grad_calculator=std_params_grad_f):
    """ Backprop function to calculate grad [dW, db] or [dW] only
    """
    def param_grad_i(optimizer, to_avg):
        def calculate_param_grad(dZ, param_grad, cache, parameters):
            (A_prev, _), next_cache = cache

            param_grad = grad_calculator(dZ, A_prev, to_avg)

            return dZ, param_grad, cache, parameters
        
        return calculate_param_grad
    return param_grad_i

