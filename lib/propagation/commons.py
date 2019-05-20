def forward_propagation(forwards, has_cache=False):
    """ Runs forward prop
    """
    def forward_prop(X, parameters):
        cache = None

        for forward in forwards:
            X, parameters, cache = forward(X, parameters, has_cache, cache)
        
        return X, cache

    return forward_prop

def backward_propagation(backwards, loss):    
    """ Runs back prop
    """
    def f(AL, Y, cache):
        parameters = None
        param_grad = None

        dZ = loss.grad_loss(AL, Y)

        for backward in reversed(backwards):
            # print('Backward name: ', backward.__name__)
            dZ, param_grad, cache, parameters = backward(dZ, param_grad, cache, parameters)

        return parameters

    return f


""" Constructing backwards
"""
def construct_backwards(backwards, optimizer, to_avg):
    """ initialize backprops functions to run
        removes first and last functions in a list
    """

    new_backwards = []

    for i in range(1, len(backwards) - 1): #disinclude first and last backwards from list
        new_backwards.append(backwards[i](optimizer, to_avg))

    return backward_propagation(new_backwards, optimizer.loss)
