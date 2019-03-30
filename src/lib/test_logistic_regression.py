import torch as pt
import unittest
from logistic_regression import *
from commons import *

device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

class TestLogisticRegression(unittest.TestCase):

    def test_initialize_params(self):
        m = 10
        n = 5
        X = pt.randn(n, m, device=device, dtype=pt.double)
        # print(X)
        w, b = initialize_params(X)

        expected_w = pt.tensor([[0], [0], [0], [0], [0]], device=device).double()
        expected_b = pt.tensor(0, device=device).double()
        
        self.assertTrue(w.equal(expected_w))
        self.assertTrue(b.equal(expected_b))
    
    # def test_propagate(self):
    #     w = pt.tensor([[1.], [2.]])
    #     b = pt.tensor([[2.]])
    #     X = pt.tensor([[1.,2.,-1.],[3.,4.,-3.2]])
    #     Y = pt.tensor([[1.,0.,1.]])

    #     # print("TEST")
    #     # print(Y.type())

    #     e_dw = pt.tensor([[ 0.99845601], [ 2.39507239]])
    #     # e_db = pt.tensor(0.00145557813678)
    #     e_cost = 5.801545319394553

    #     grads, cost = propagate(w, b, X, Y)

    #     dw, db = grads["dw"], grads["db"]

    #     # print("dw: {}".format(dw))
    #     # print("db: {}".format(db))
    #     # print(db)
    #     # print(e_db)

    #     self.assertTrue(dw.equal(e_dw))
    #     # self.assertTrue(db.equal(e_db))
    #     self.assertEqual(cost, e_cost)
    
    # def test_optimize(self):
    #     w = pt.tensor([[1.], [2.]])
    #     b = pt.tensor([[2.]])
    #     X = pt.tensor([[1.,2.,-1.],[3.,4.,-3.2]])
    #     Y = pt.tensor([[1.,0.,1.]])

    #     params, costs = optimize(w, b, X, Y, iterations=100, learning_rate=0.009, is_printable_cost=False)

    #     print(params)
    #     print(costs)

    def test_predict(self):
        w = pt.tensor([[0.1124579],[0.23106775]])
        b = -0.3
        X = pt.tensor([[1.,-1.1,-3.2],[1.2,2.,0.1]])

        print("Y pred: ")
        print(predict(w, b, X))

if __name__ == "__main__":
    unittest.main()
