import torch as pt
import lib.logistic_regression as lr
from utils.loader import *
from utils.data_divider import *
from utils.plot_cost import *
import matplotlib.pyplot as plt

device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

banknote = loader("banknote.csv", device)

# X = banknote["X"]
# Y = banknote["Y"]

# print(X.shape)
# print(Y.shape)
# print(banknote[])

X_train, Y_train, X_test, Y_test = divide_data(banknote)

# print("train shape")
# print(X_train.shape)
# print(Y_train.shape)

# bndict = lr.model(X_train, Y_train, X_test, Y_test, is_print_cost=True)

# print(bndict)
# plot_cost(bndict)

learning_rates = [0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = lr.model(X_train, Y_train, X_test, Y_test, learning_rate=i)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(models[str(i)]["costs"], label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()