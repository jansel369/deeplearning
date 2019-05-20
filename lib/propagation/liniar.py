
def liniar_forward(A_prev, params, has_cache, cache):
    """cache format
        ((A_prev, [W, b]), (((**other cahche), [W, b]),...))
    """

    current_params, next_params = params
    W, b = current_params

    Z = A_prev.mm(W) + b

    cache = ((A_prev, current_params), cache) if has_cache else None

    return Z, next_params, cache

def update_param_a():
    """ Backprop function initialized by specified optimizer update
    """
    def construct_update(optimizer, to_avg):
        return optimizer.param_update_f(optimizer, to_avg)

    return construct_update

def liniar_backward_a(): # called from architecture
    """ calcluates grad dL/da
    """
    def liniar_backward_i(optimizer, to_avg): # called from back prop initialization
        def liniar_backward(dZ, param_grad, cache, parameters): # called during backprop
            (_, [W, _]), next_cache = cache

            dA = dZ.mm(W.t())

            return dA, param_grad, cache, parameters

        return liniar_backward
    
    return liniar_backward_i

def activation_forward_a(activation):
    def activation_forward(Z, params, has_cache, cache):
        A = activation(Z)

        return A, params, cache

    return activation_forward

def liniar_grad_f(activation_backward):
    """ Backprop function to calculate grad dL/dz
    """
    def liniar_grad_i(optimizer, to_avg):
        def liniar_grad(dA, param_grad, cache, parameters):
            current_cache, next_cache = cache
            A, _ = current_cache

            dZ = dA * activation_backward(A)
            
            return dZ, param_grad, next_cache, parameters
        
        return liniar_grad
    return liniar_grad_i



""" Helper functions: Calculating gradient parameters dW, db for liniar and bn
"""

def _weight_grad(dZ, avg, A_prev):
    return avg * A_prev.t().mm(dZ)

def _bias_grad(dZ, avg, fdim=0):
    return avg * dZ.sum(dim=fdim, keepdim=True)

def std_params_grad_f(dZ, A_prev, to_avg):
    dW = _weight_grad(dZ, to_avg, A_prev)
    db = _bias_grad(dZ, to_avg)

    return [dW, db]
    
def bn_prams_grad_f(dZ, A_prev, to_avg):
    dW = _weight_grad(dZ, to_avg, A_prev)

    return [dW]

def liniar_param_grad_a(select_grad=std_params_grad_f):
    """ Backprop function to calculate grad [dW, db] or [dW] only
    """
    def liniar_param_grad_i(optimizer, to_avg):
        def calculate_liniar_param_grad(dZ, param_grad, cache, parameters):
            (A_prev, _), next_cache = cache

            param_grad = select_grad(dZ, A_prev, to_avg)

            return dZ, param_grad, cache, parameters
        
        return calculate_liniar_param_grad
    return liniar_param_grad_i

