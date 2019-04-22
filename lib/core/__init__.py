# from .activation import *
# from .loss import *
# from .propagation import *

from .core import *
from .list import *

from . import list
from . import core

from .core import binary_crossentropy_backward, categorical_crossentoropy_backward as gradient
from .core import binary_crossentropy_cost, categorical_crossentropy_cost as cost
from .core import sigmoid_forward, relu_forward, softmax_forward as activation_forward
from .core import sigmoid_backward, relu_backward as activation_backward
from .core import liniar_forward, activation_forward as propagation
from .core import predict
from .core import categorical_crossentoropy_predict_accuracy, binary_crossentropy_predict_accuracy as predict_accuracy

from .list import liniar, sigmoid, tanh, relu as activation
from .list import categorical_crossentropy, binary_crossentropy as loss