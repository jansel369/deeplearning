from . import propagation as p
# from . import loss as l

def categorical_crossentoropy_predict_accuracy(AL, Y):
    equality = AL.argmax(0).eq(Y.argmax(0))

    return equality.float().mean() * 100

def binary_crossentropy_predict_accuracy(AL, Y):

    return 100 - (AL - Y).abs().double().mean() * 100

# predict_accuracy_dict = {
#     l.binary_crossentropy: binary_crossentropy_predict_accuracy,
#     l.categorical_crossentropy: categorical_crossentoropy_predict_accuracy,
# }