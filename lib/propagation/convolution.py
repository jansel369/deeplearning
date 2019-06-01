import torch as pt
from torch.nn.functional import pad
from .liniar import liniar_forward

def _zero_pad(A, p):
    return pad(A, (0, 0, p, p, p, p), 'constant', 0.)

def _A_slice(A, h, w, s, f):
    h_start = h * s # vertical
    h_end = h_start + f
    w_start = w * s # horizontal
    w_end = w_start + f

    return A[:, h_start:h_end, w_start:w_end, :] # a_slice of A_prev

def _conv_single_step(a_slice, W, b):
    return pt.sum((a_slice * W), [1, 2, 3]) + b.sum()

""" CONVOLUTION
"""

def conv_forward_a(p, s, n_C):
    def conv_forward(A_prev, params, has_cache, cache):
        """ W shape: (f, f, n_C_prev, n_C)
            b shape: (1, 1, 1, n_C)
        """

        current_params, next_params = params
        W, b = current_params
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        f, f, n_C_prev, n_C = W.shape

        n_H = int(( n_H_prev - f + 2 * p ) / s) + 1
        n_W = int(( n_W_prev - f + 2 * p ) / s) + 1

        # initialize output volume
        Z = pt.zeros((m, n_H, n_W, n_C), device=A_prev.device, dtype=pt.double)
        
        A_prev_pad = _zero_pad(A_prev, p)

        # for i in range(m):  
        #     a_prev_pad = A_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                a_prev_slice = _A_slice(A_prev_pad, h, w, s, f) # a_slice of A_prev

                for c in range(n_C): # loop over the number of filters
                    Z[:, h, w, c] = _conv_single_step(a_prev_slice, W[:, :, :, c], b[:, :, :, c])

        cache = (((A_prev, p, s, n_C), current_params), cache) if has_cache else None

        return Z, next_params, cache
    return conv_forward

def conv_backward_a():
    """ Backprop to calculate dL/dA
    """
    def conv_backward_i(optimizer, to_avg):
        def conv_backward(dZ, param_grad, cache, parameters):
            current_cache, next_cache = cache
            (A_prev, p, s, n_C), [W, b] = current_cache
            m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
            m, n_H, n_W, n_C = dZ.shape
            f, f, n_C_prev, n_C = W.shape
            dtype=pt.double
            device = A_prev.device

            dA = pt.zeros((m, n_H_prev, n_W_prev, n_C_prev), dtype=dtype, device=device)  
            dA_pad = _zero_pad(dA, p)

            # for i in range(m):
            #     da_pad = dA_pad[i]

            for h in range(n_H):
                for w in range(n_W):
                    h_start = h * s # vertical
                    h_end = h_start + f
                    w_start = w * s # horizontal
                    w_end = w_start + f

                    for c in range(n_C):
                        # print("dZ: ", dZ.shape)
                        weight = pt.ones((m, f * f * n_C_prev), dtype=dtype, device=device) * W[:, :, :, c].view(-1)
                        dz = dZ[:, h, w, c].view((m, 1))
                        # da = (weight * dz).view((m, f, f, n_C_prev))
                        # print("shape: w, dz", weight.shape, dz.shape)

                        dA_pad[:, h_start:h_end, w_start:w_end, :] += (weight * dz).view((m, f, f, n_C_prev))

                dA[:, :, :, :] = dA_pad[:, p:n_H_prev+p, p:n_W_prev+p, :]

            return dA, param_grad, cache, parameters
        
        return conv_backward
    return conv_backward_i



""" Helper functions: Calculating gradient parameters dW, db for conv and bn
"""
def std_param_grad_f(dz):
    return dz

def bn_param_grad_f(ddz):
    return 0

def conv_param_grad_a(select_grad=std_param_grad_f):
    """ Backprop to calcualte dL/dW and dL/db
    """
    def conv_param_grad_f(optimizer, to_avg):
        def conv_param_grad(dZ, param_grad, cache, parameters):
            current_cache, _ = cache
            (A_prev, p, s, n_C), [W, b] = current_cache
            m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
            m, n_H, n_W, n_C = dZ.shape
            f, f, n_C_prev, n_C = W.shape

            dW = pt.zeros((f, f, n_C_prev, n_C), dtype=pt.double, device=A_prev.device)
            db = pt.zeros((1, 1, 1, n_C), dtype=pt.double, device=A_prev.device)
            A_prev_pad = _zero_pad(A_prev, p)

            for h in range(n_H):
                for w in range(n_W):
                    a_prev_slice = _A_slice(A_prev_pad, h, w, s, f)

                    for c in range(n_C):
                        dz = dZ[:, h, w, c]
                        dW[:,:,:,c] += pt.tensordot(dz, a_prev_slice, 1) # multiply 1d(dz) by 4d(a_prev_slice) and sum by 1st dim together
                        db[:,:,:,c] += select_grad(dz).sum(0)

            return dZ, [dW * to_avg, db * to_avg], cache, parameters
        
        return conv_param_grad
    return conv_param_grad_f

def conv_grad_a(activation_backward):
    """ Backprop calculating grad dL/dZ
    """
    def conv_grad_i(optimizer, to_avg):
        def conv_grad(dA, param_grad, cache, parameters):
            current_cache, next_cache = cache
            A, _, _= current_cache

            dZ = dA * activation_backward(A)

            return dZ, param_grad, next_cache, parameters
        return conv_grad
    return conv_grad_i


""" POOLING
"""

def max_pool(a):
    m, n_H, n_W = a.shape

    return a.reshape(m, -1).max(1)[0]

def avg_pool(a):
    return pt.mean(a, [1, 2])

def max_pool_forward_a(f, s):
    return pool_forward_a(f, s, max_pool)

def avg_pool_forward_a(f, s):
    return pool_forward_a(f, s, avg_pool)

def pool_forward_a(f, s, pool=max_pool):
    def pool_forward(A_prev, params, has_cache, cache):
        m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
        n_H = int(1 + (n_H_prev - f) / s)
        n_W = int(1 + (n_W_prev - f) / s)
        n_C = n_C_prev

        A = pt.zeros((m, n_H, n_W, n_C), dtype=pt.double, device=A_prev.device)

        # for i in range(m):
        #     a_prev_pad = A_prev[i]

        for h in range(n_H):
            for w in range(n_W):
                a_prev_slice = _A_slice(A_prev, h, w, s, f)

                for c in range(n_C):
                    A[:, h, w, c] = pool(a_prev_slice[:, :, :, c])
        
        cache = ((A_prev, f, s), cache) if has_cache else None

        return A, params, cache
    return pool_forward

def _max_pool_mask(da, slice, device):
    m, n_H, n_W = slice.shape
    slice_2d = slice.reshape((m, -1))
    slice_max = slice_2d.max(1, keepdim=True)[0]

    return (slice_2d == slice_max).double().view((m, n_H, n_W))

def _avg_value_distribute(da, slice, device):
    m, n_H, n_W = slice.shape
    avg = da / (n_H + n_W)

    slice2d = pt.full((m, n_H * n_W), 1.0, dtype=pt.double, device=device) * avg.view((m, 1))

    return slice2d.view(m, n_H, n_W)

def _pool_backward_a(type_pool_backward):
    def pool_backward_i(optimizer, to_avg):
        def pool_backward(dA, param_grad, cache, parameters):
            _, current_cache = cache
            (A_prev, f, s), next_cache = current_cache
            m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
            m, n_H, n_W, n_C = dA.shape
            device = A_prev.device

            dA_prev = pt.zeros((m, n_H_prev, n_W_prev, n_C_prev), dtype=pt.double, device=A_prev.device)

            # for i in range(m):
            #     a_prev = A_prev[i]

            for h in range(n_H):
                for w in range(n_W):
                    h_start = h * s
                    h_end = h_start + f
                    w_start = w * s
                    w_end = w_start + f

                    for c in range(n_C):
                        a_prev_slice = A_prev[:, h_start:h_end, w_start:w_end, c]
                        da = dA[:, h, w, c]

                        dA_prev[:, h_start:h_end, w_start:w_end, c] += type_pool_backward(da, a_prev_slice, device)

            return dA_prev, param_grad, current_cache, parameters
        return pool_backward
    return pool_backward_i

def max_pool_backward_a():
    return _pool_backward_a(_max_pool_mask)

def avg_pool_backward_a():
    return _pool_backward_a(_avg_value_distribute)

""" Flatting
"""

def flatten_forward(A_prev, params, has_cache, cache):
    shape = A_prev.shape

    A_prev_flat = A_prev.reshape(shape[0], -1)

    # cache = (shape, ((A_prev, None, None, None), cache)) if has_cache else None
    cache = (shape, cache) if has_cache else None

    return A_prev_flat, params, cache


# fully_connected = liniar_forward


def flatten_backward_i(optimizer, to_avg): 
    def flatten_backward(dA, param_grad, cache, parameters):
        _, current_cache = cache
        shape, _ = current_cache

        dA_vol = dA.reshape(shape)

        return dA_vol, param_grad, current_cache, parameters
    return flatten_backward
