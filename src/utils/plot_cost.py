import matplotlib.pyplot as plt

def plot_cost(dict):
    plt.plot(dict["costs"])
    plt.ylabel("costs")
    plt.xlabel("iterations / 100s")
    plt.title("Logistic Regression (a=" + str(dict["learning_rate"]) + ")")
    plt.show()

