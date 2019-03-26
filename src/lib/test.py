import torch as pt
from commons import *

dtype = pt.float
cuda0 = pt.device("cuda:0")

# A = pt.tensor([[1., 2., 3.], [4., 5., 6.]], device=cuda0)
# B = pt.tensor([[2.], [4.], [6.]], device=cuda0)
# print(A.shape)
# print(B.shape)
# print(sigmoid(A.mm(B)).device)

