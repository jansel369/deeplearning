import torch as pt
import matplotlib.pyplot as plt
from commons import *

def initialize_params(X):

    w = pt.zeros(X.shape[0], 1, device=X.device, dtype=pt.double)
    b = pt.tensor(0, device=X.device, dtype=pt.double)

    return w, b

def propagate(w, b, X, Y, has_cost=True):
    # w - wight vector (n, 1)
    # b - bias integer
    # X - input matrix (n, m)
    # Y - output label (1, m)
    m = X.shape[1]
    
    # forward progaragtion
    A = sigmoid(w.t().mm(X) + b)
    print(A)
    cost =  - (1 / m) * (Y * pt.log(A) + (1 - Y) * pt.log(1 - A)).sum().item() if has_cost else 0

    # backward propagation
    dz = A - Y
    dw = (1 / m) * X.mm(dz.t())
    db = (1 / m) * dz.sum()

    grads = { "dw": dw, "db": db }

    return grads, cost

def optimize(w, b, X, Y, iterations, learning_rate, is_printable_cost):
    costs = []

    for i in range(iterations):

        has_cost = i % 100 == 0

        grads, cost = propagate(w, b, X, Y, has_cost)

        dw = grads["dw"]
        db = grads["db"]

        w -= learning_rate * dw
        b -= learning_rate * db

        if has_cost:
            costs.append(cost)
            if is_printable_cost:
                print("Cost after iteration %i: %f" %(i, cost))
    
    params = { "w": w, "b": b }

    return params, costs

def predict(w, b, X, threshold = 0.5):

    A = sigmoid(w.t().mm(X) + b)
    Y_pred = A > threshold

    return Y_pred

def model(X_train, Y_train, X_test, Y_test, iterations=2000, learning_rate=0.05, prediction_threshold=0.5, is_print_cost=False):
    n, m = X_train.shape

    w, b = initialize_params(X_train)
    parameters, costs = optimize(w, b, X_train, Y_train, iterations, learning_rate, is_print_cost)

    w = parameters["w"]
    b = parameters["b"]

    prediction_threshold = 0.5
    Y_pred_train = predict(w, b, X_train, prediction_threshold)
    Y_pred_test = predict(w, b, X_test, prediction_threshold)

    print("train accuracy: {} %".format(100 - (Y_pred_train - Y_train).abs().double().mean() * 100))
    print("test accuracy: {} %".format(100 - (Y_pred_test - Y_test).abs().double().mean() * 100))

    plt.plot(costs)
    plt.ylabel("costs")
    plt.xlabel("iterations / 100s")
    plt.tile("Logistic Regression (a=" + learning_rate + ")")
    plt.show()

    return {
        "costs": costs,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "iterations": iterations,
        "y_pred_test": Y_pred_test,
        "y_pred_train": Y_pred_train
    }
