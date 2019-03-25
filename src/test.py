import torch as pt
import lib.logistic_regression as lr
from utils.loader import *
from utils.data_divider import *

device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

banknote = loader("banknote.csv", device)

X_train, Y_train, X_test, Y_test = divide_data(banknote)

lr.model(X_train, Y_train, X_test, Y_test, is_print_cost=True)

print(banknote["Y"])
print(banknote["X"])