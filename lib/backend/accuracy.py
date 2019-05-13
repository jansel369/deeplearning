
def categorical_crossentoropy_predict_accuracy(AL, Y):
    # get the maximum value feature in every row: (m, n) -> (m,1)
    # compare both result with values: 1/0
    equality = AL.argmax(1).eq(Y.argmax(1))

    return equality.double().mean() * 100

def binary_crossentropy_predict_accuracy(AL, Y):

    return 100 - (AL - Y).abs().double().mean() * 100

