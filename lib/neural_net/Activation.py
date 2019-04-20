import copy

linear = 'linear'
sigmoid = 'sigmoid'
tanh = 'tanh'
relu = 'relu'
softmax = 'softmax'


def Relu():
    def f(config):
        config = copy.deepcopy(config)

        config['layers'][-1]['activation'] = 'relu'

        return config

    return f

def Sigmoid():
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activation'] = 'sigmoid'

        return config

    return f

def Softmax():
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activation'] = 'softmax'
        
        return config
    
    return f

