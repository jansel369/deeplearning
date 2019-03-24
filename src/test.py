import torch as pt
import lib.logistic_regression as lr
from utils.loader import *

device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

banknote = loader("banknote.csv", device)

print(banknote["Y"])
print(banknote["X"])