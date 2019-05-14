def activation_forward_a(activation):
    def activation_forward(Z, params, has_cache, cache):
        A = activation(Z)

        return A, params, cache

    return activation_forward
