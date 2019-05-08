from collections import namedtuple
from . import gradient
from . import accuracy
from . import cost

Loss = namedtuple('Loss', 'loss, grad_loss, pred_acc, compute_cost')

categorical_crossentropy = Loss(
    'categorical_crossentropy',
    gradient.categorical_crossentoropy_backward,
    accuracy.categorical_crossentoropy_predict_accuracy,
    cost.categorical_crossentropy_cost,
)
binary_crossentropy = Loss(
    'binary_crossentropy',
    gradient.binary_crossentropy_backward,
    accuracy.binary_crossentropy_predict_accuracy,
    cost.binary_crossentropy_cost,
)
