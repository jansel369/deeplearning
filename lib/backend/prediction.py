from . import propagation as p

def predict(X, parameters, layers):
    Al = X
    L = len(layers)

    for l in range(1, L):
        layer = layers[l]
        A_prev = Al

        Wl = parameters["W" + str(l)]
        bl = parameters["b" + str(l)]

        Zl = p.liniar_forward(A_prev, Wl, bl)
        Al = p.activation_forward(Zl, layer['activation'])
    
    return Al

def categorical_crossentoropy_predict_accuracy(X, Y, parameters, layers):
    AL = predict(X, parameters, layers)

    equality = AL.argmax(0).eq(Y.argmax(0))

    return equality.float().mean() * 100, AL

def binary_crossentropy_predict_accuracy(X, Y, parameters, layers):
    AL = predict(X, parameters, layers)

    return 100 - (AL - Y).abs().double().mean() * 100, AL
    # return AL.eq(Y).float().mean() * 100
