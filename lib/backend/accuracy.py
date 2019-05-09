
def categorical_crossentoropy_predict_accuracy(AL, Y):
    equality = AL.argmax(0).eq(Y.argmax(0))

    return equality.float().mean() * 100

def binary_crossentropy_predict_accuracy(AL, Y):

    return 100 - (AL - Y).abs().double().mean() * 100

