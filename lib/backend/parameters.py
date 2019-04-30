from . import initialization as init

batch_norm = "batch_norm"
standard = "standard"

def batch_norm_params(params, n, n_prev, l, device):
    params["beta" + l] = pt.zeros(n, 1, dtype=pt.double, device=device)
    params['gamma' + l] = pt.zeros(n, 1, dtype=pt.double, device=device)

    return params

def standard_params(params, n, n_prev, l, device):
    params["b" + l] = pt.zeros(n, 1, dtype=pt.double, device=device)

    return params

partial_params_dict = {
    standard: standard_params,
    batch_norm: batch_norm_params,
}

def init_params(layers, device=pt.device("cpu")):
    params = {}

    for l in range(len(layers) - 1):
        n = layers[l + 1]["units"]
        cur_layer = layers[l] 
        n_prev = cur_layer["units"]
        initialization = cur_layer['initialization'] 
        partial_params = cur_layer['parameter_type']

        W = pt.randn(n, n_prev, dtype=pt.double, device=device)

        l_s = str(l+1)
        params["W" + l_s] = init.init_dict[initialization](W, n, n_prev)

        params = partial_params_dict[partial_params](params, n, n_prev, l_s, device)

    return params

def batch_norm_update(l, params, grads, learning_rate):
    params['beta' + l] -= learning_rate * grads['dbeta' + l]
    params['gamma' + l] -= learning_rate * grads['dgamma' + l]

    return params

def standard_update(l, params, grads, learning_rate):
    params['db' + l] -= learning_rate * grads['db' + l]

    return params

update_update_dict = {
    standard: standard_update,
    batch_norm: batch_norm_update,
}

def update_parameters(layers, parameters, grads, learning_rate):
    for l in range(1, len(layers)):
        l_s = str(l)
        partial_params = layers[l]['parameter_type']

        parameters["W" + l_s] -= learning_rate * grads["dW" + l_s]

        parameters = update_update_dict[partial_params](l_s, parameters, grads, learning_rate)
    
    return parameters
