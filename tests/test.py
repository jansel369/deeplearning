import sys
print(sys.path[1])
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../lib")

print(sys.path)

import neural_net as test

test.test()

# welcome to dongmakgol
# import lib

# import lib.test

# dir(lib)

# print(lib)


# print(__name__)


# test.say()

# say()