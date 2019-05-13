from torch.nn.functional import pad
from .liniar import liniar_forward

def conv_single_step(a_slice, W, b):
    return (a_slice * W).sum() + b.sum()

def conv_forward_a(p, s, n_C):
    def conv_forward(A_prev, params, has_cache, cache):

        current_params, next_params = params
        W, b = current_params

        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        f, f, n_C_prev, n_C = W.shape

        n_H = int(( n_H_prev - f + 2 * p ) / s) + 1
        n_W = int(( n_W_prev - f + 2 * p ) / s) + 1

        # initialize output volume
        Z = pt.zeros((m, n_H, n_W, n_C), device=A_prev.device)
        pad_size = (0, 0, p, p, p, p)
        A_prev_pad = pad(A_prev, pad_size, 'constant', 0.)

        for i in range(m):
            
            a_prev_pad = A_prev_pad[i]

            for h in range(n_H):
                for w in range(n_W):

                    h_start = h * s # vertical
                    h_end = h_start + f
                    w_start = w * s # horizontal
                    w_end = w_start + f

                    a_prev_slice = a_prev_pad[h_start:h_end, w_start:w_end, :] # a_slice of A_prev
                  
                    for c in range(n_C): # loop over the number of filters
                        Z[i, h, w, c] = conv_single_step(a_prev_slice, W[:, :, :, c], b[:, :, :, c])

        cace = (((A_prev, p, s, n_C), current_params), cache) if has_cache else None

        return Z, next_params, cache



def max_pool(a):
    return a.max()

def avg_pool(a):
    return a.mean()

def pool_forward_a(f, s, pool=max_pool):
    def pool_forward(A_prev, params, has_cache, cache):
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        n_H = int(1 + (n_H_prev - f) / s)
        n_W = int(1 + (n_W_prev - f) / s)
        n_C = n_C_prev

        A = pt.zeros((m, n_H, n_W, n_C), device=A_prev.device)

        for i in range(m):
            
            a_prev_pad = A_prev[i]

            for h in range(n_H):
                for w in range(n_W):
                    h_start = h * s # vertical
                    h_end = h_start + f
                    w_start = w * s # horizontal
                    w_end = w_start + f

                    a_prev_slice = a_prev_pad[h_start:h_end, w_start:w_end, :]

                    for c in range(n_C):
                        A[i, h, w, c] = pool(a_prev_slice[:, :, c])
        
        cache = ((f, s, pool),cache) if has_cache else None

        return A, params, cache

    return pool_forward

def flatten_forward(A_prev, params, has_cache, cache):
    shape = A_prev.shape


    A_prev_flat = A_prev.reshape(shape[0], -1).t()

    cache = (shape, cache) if has_cache else None

    return A_prev_flat, params, cache


fully_connected = liniar_forward

def flatten_backward(dA, param_grad, cache, parameters):
    shape, next_cache = cache

    dA_vol = dA.t().reshape(shape)

    return dA_vol, param_grad, next_cache, parameters
