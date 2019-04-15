import copy

def Relu():
    def f(config):
        config = copy.deepcopy(config)

        config['layers'][-1]['activations'].append("relu")

        return config

    return f

def Sigmoid():
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activations'].append("sigmoid")

        return config

    return f

def Softmax():
    def f(config):
        config = copy.deepcopy(config)
        config['layers'][-1]['activations'].append("softmax")
        
        return config
    
    return f

