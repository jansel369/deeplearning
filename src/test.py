# import sys
# print(sys.path)
# from utils.device_agnostic import *
import torch as pt
from utils.loader import *

device = pt.device("cuda") if pt.cuda.is_available() else pt.device("cpu")

banknote_dict = loader("banknote.csv", device)
print(banknote_dict["headers"])
print(banknote_dict["m"])
