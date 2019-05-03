from backend import loss as l
from backend import activation as a
from backend import cost as c
from backend import gradient as g
from backend import propagation as p
from backend import prediction as pred
from . import commons

def forward_propagation(forwards, has_cache=False):
    def f(X, parameters):
        cache = None

        for forward in forwards:
            X, parameters, cache = forward(X, parameters, has_cache, cache)
        
        return X, cache

    return f

""" constructing backward prop functions
    naming conventins: (activation + '_')<optimizer>_<type>_<method>
"""

def construct_backwards(layers, learnibng_rate, m):
    """ temporary constructor
        todo: batch norm
    """
    backwards = []
    backward_dict = { # for caching purposes to avoid creation redundancy
        'gr_std_update': gr_std_update(learnibng_rate, m),
    }

    backwards.append(backward_dict['gr_std_update'])

    for l in reversed(range(1, len(layers) - 1)):
        layer = layers[l]
        activation = layer['activation']

        liniar_fn = activation + '_liniar_std_grad'

        # add liniar grad calculation
        if liniar_fn in backward_dict: # use cached function
            backwards.append(backward_dict[liniar_fn])
        else: # cache new function
            backward = liniar_std_grad(a.activations_dict[activation])
            backwards.append(backward)
            backward_dict[liniar_fn] = backward

        update_fn = 'gr_std_update'
        backwards.append(backward_dict[update_fn])

    return backwards


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

def gr_std_update(learning_rate, m):

    weight_grad = weight_std_grad(m)
    bias_grad = bias_std_grad(m)

    def update(dZ, cache, parameters):
        """Gradient Descent standard update
            cache - tuple cache: ((A_prev, W, b), ((next_cache, ...)))
            parameters - tuple updated parameters: ((W, b), ((prev_params, ...)))
        """
        current_cache, next_cache = cache
        A_prev, W, b = current_cache

        dW = weight_grad(dZ, A_prev)
        db = bias_grad(dZ)

        W -= learning_rate * dW
        b -= learning_rate * db

        return dZ, cache, ((W, b), parameters)

    return update

""" activatoin grad
"""

# def activation_grad(dZ, cache, parameters):
#     current_cache, next_cache = cache
#     A_prev, W, b = current_cache

#     dA = W.t().mm(dZ)

#     return dA, cache, parameters

""" liniar grad
"""

def liniar_std_grad(activation_backward):
    def f(dZ, cache, parameters):
        current_cache, next_cache = cache
        A, W_preced, b_preced = current_cache

        dZ = W_preced.t().mm(dZ) * activation_backward(A)

        return dZ, next_cache, parameters

    return f

""" Backward propagation
"""

def backward_propagation(backwards, loss):
    grad_loss = g.loss_backward_dict[loss]
    
    def f(AL, Y, cache):
        dZ = grad_loss(AL, Y)
        parameters = (None)

        for backward in backwards:
            dZ, cache, parameters = backward(dZ, cache, parameters)

        return parameters

    return f

def gradient_descent(learning_rate, iterations, loss):
    compute_cost = c.costs_dict[loss]

    def optimizer(X, Y, parameters, config, is_printable_cost):
        costs = []
        layers = config['layers']
        L = len(layers)
        m = Y.shape[1]

        forwards = config['forwards']
        print(forwards)
        backwards = construct_backwards(layers, learning_rate, m)

        forward_prop = forward_propagation(forwards, has_cache=True)
        backward_prop = backward_propagation(backwards, loss)

        for i in range(iterations):
            has_cost = i % 100 == 0

            AL, cache = forward_prop(X, parameters)

            if has_cost:
                cost = compute_cost(AL, Y)
                costs.append(cost)

                if is_printable_cost:
                    print("Cost after iteration %i: %f " % (i, cost))

            parameters = backward_prop(AL, Y, cache)

        return parameters, costs

    return optimizer