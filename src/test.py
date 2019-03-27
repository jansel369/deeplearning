import torch as pt
import lib.logistic_regression as lr
from utils.loader import *
from utils.data_divider import *

device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

banknote = loader("banknote.csv", device)

# X = banknote["X"]
# Y = banknote["Y"]

# print(X.shape)
# print(Y.shape)
# print(banknote[])

X_train, Y_train, X_test, Y_test = divide_data(banknote)

print("train shape")
print(X_train.shape)
print(Y_train.shape)

dict = lr.model(X_train, Y_train, X_test, Y_test, is_print_cost=True)

print(dict)