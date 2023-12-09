
# ACTIVATION FUNCTION

# Step Function
# y = 1  x > 0
# y = 0  x <= 0

# Sigmoid function

# ReLU function - Rectified Linear Unit
'''
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []


for i in inputs:
    output.append(max(0, i))

print(output)
'''
'''
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output)
'''

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)


class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # valores mais ou menos entre -1 e 1
        self.biases = np.zeros((1, n_neurons)) # bias iniciado com zeros
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # não precisa mais do transpose



class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)
#print(layer1.output) # aqui aparecem alguns números negativos
activation1.forward(layer1.output)
print(activation1.output) # os numeros negativos viraram zeros