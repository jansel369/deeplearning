def activation_forward_a(activation):
    def activation_forward(Z, params, has_cache, cache):
        A = activation(Z)

        return A, params, cache

    return activation_forward

def activation_grad_a(): # called from architecture
    """ calcluates grad dL/da
    """

    def activation_grad_f2(optimizer, to_avg): # called from back prop initialization
        def activation_grad(dZ, param_grad, cache, parameters): # called during backprop
            (_, [W, _]), next_cache = cache

            dA = W.t().mm(dZ)

            return dA, param_grad, cache, parameters

        return activation_grad
    
    return activation_grad_f2
 