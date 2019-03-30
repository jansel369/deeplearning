import torch as pt
import unittest
from logistic_regression import *
from commons import *

device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

class TestLogisticRegression(unittest.TestCase):

    def test_sigmoid(self):
        z = pt.tensor([0., 2.])

        a = sigmoid(z)

        e_a = pt.tensor([0.5, 0.8807970285415649])

        # print(a[0], a[1].item())

        self.assertTrue(a.equal(e_a))

if __name__ == "__main__":
    unittest.main()