from collections import namedtuple
from . import gradient

Loss = namedtuple('Loss', 'loss, grad_loss')

categorical_crossentropy = Loss('categorical_crossentropy', gradient.categorical_crossentoropy_backward)
binary_crossentropy = Loss('binary_crossentropy', gradient.binary_crossentropy_backward)
