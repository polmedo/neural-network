import numpy as np

class Neuron:
    def__init__(self, num_inputs):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    