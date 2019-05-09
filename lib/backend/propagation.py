# from .activation import activations_dict, activation_backward_dict

# def liniar_forward(A_prev, W, b):
#     Z = W.mm(A_prev) + b

#     return Z

# def activation_forward(Z, activation):
#     A = activations_dict[activation](Z)

#     return A

# def forward_propagation(X, parameters, layers):

#     caches = []
#     Al = X
#     L = len(layers)
    
#     for l in range(1, L):

#         layer = layers[l]
#         A_prev = Al

#         Wl = parameters["W" + str(l)]
#         bl = parameters["b" + str(l)]

#         Zl = liniar_forward(A_prev, Wl, bl)
#         Al = activation_forward(Zl, layer['activation'])

#         caches.append((A_prev, Wl))

#     return Al, caches
    

# def backward_propagation(dZL, caches, layers):
#     grads = {}
#     dZl = dZL
#     # L = len(layers)
#     m = dZL.shape[1]

#     for l in reversed(range(len(caches))):

#         # print("l: ", l)
#         (A_prev, Wl) = caches[l]

#         grads['dW' + str(l + 1)] =  (1 / m) * dZl.mm(A_prev.t())
#         grads['db' + str(l + 1)] = (1 / m) * dZl.sum(dim=1, keepdim=True)

#         if l > 0:
#             prev_activation = layers[l]['activation']
#             # print('prev activation: ', l,  prev_activation)
#             dZl = Wl.t().mm(dZl) * activation_backward_dict[prev_activation](A_prev)

#     return grads
