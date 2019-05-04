from backend import cost as c
import matplotlib.pyplot as plt

class CostEvaluator:
    """ cost evaluator manages and display cost result during training
    """

    def __init__(self, loss, learning_rate, alternation=100, print_cost=False):
        self.costs = []
        self.alternation = alternation
        self.compute_cost = c.costs_dict[loss]
        self.print_cost = print_cost
        self.learning_rate = learning_rate

    def add_cost(self, count, AL, Y):
        if count % self.alternation != 0:
            return
        
        cost = self.compute_cost(AL, Y)
        self.costs.append(cost)

        if self.print_cost:
            print("Cost after iteration %i: %f " % (count, cost))

    def plot_cost(self):
        plt.plot(self.costs)
        plt.ylabel("costs")
        plt.xlabel("iterations / 100s")
        plt.title("neural net (a=" + str(self.learning_rate) + ")")
        plt.show()
