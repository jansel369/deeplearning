
def divide_data(dict, percentage = 0.2):
    X = dict["X"]
    Y = dict["Y"]
    m = dict["m"]
    test_count = round(m * percentage)

    X_test = X[0:test_count, :]
    Y_test = Y[:, 0:test_count].t()
    X_train = X[test_count:m, :]
    Y_train = Y[:, test_count:m].t()

    return X_train, Y_train, X_test, Y_test