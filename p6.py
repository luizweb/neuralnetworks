import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # valores mais ou menos entre -1 e 1
        self.biases = np.zeros((1, n_neurons)) # bias iniciado com zeros
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # não precisa mais do transpose



class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)



class Activation_Softmax:
    def forward(self, inputs): 
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # evitar overflow com o 'max'
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities



# Batch
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])



# ---------------------------------------------------------

# Softmax Activation
# Batch

"""
import numpy as np 
import nnfs

nnfs.init()


layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]



exp_values = np.exp(layer_outputs)

#print(np.sum(layer_outputs, axis=1, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
"""


# ---------------------------------------------------------

# Softmax Activation
# Mesma lógica do código comentado abaixo, porém com numpy

# input >> exponentiate >> normalize >> output
# Softmax (exponentiate >> normalize)


"""

import numpy as np 


layer_outputs = [4.8, 1.21, 2.385]


exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values)) 
"""

# ----------------------------------------------------------

"""
# Softmax Activation

import math

layer_outputs = [4.8, 1.21, 2.385]

# número Euler
# E = 2.718281828459045
E = math.e

# exponentials
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)

print(exp_values)

# normalization

norm_base = sum(exp_values)
norm_values = []

for values in exp_values:
    norm_values.append( values / norm_base )

print(norm_values)
print(sum(norm_values)) # resultado 1 ou muito próximo a 1
"""