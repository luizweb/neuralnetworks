
import numpy as np

np.random.seed(0)


X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0,2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # valores mais ou menos entre -1 e 1
        self.biases = np.zeros((1, n_neurons)) # bias iniciado com zeros
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases # não precisa mais do transpose

layer1 = Layer_Dense(4,3)
layer2 = Layer_Dense(3,2)


layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)





# print(0.10 * np.random.randn(4, 3))
# print(np.zeros((1, 3)))
'''
[[ 0.17640523  0.04001572  0.0978738 ]
 [ 0.22408932  0.1867558  -0.09772779]
 [ 0.09500884 -0.01513572 -0.01032189]
 [ 0.04105985  0.01440436  0.14542735]]

[[0. 0. 0.]]
'''


# ----------------------------------------------------------------------------------

'''
# ADD LAYER

import numpy as np


inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0,2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases 
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2 

print(layer1_outputs)
'''

# -----------------------------------------------------------------------------

'''
# BATCH INPUTS

import numpy as np

# Vector (4,)
# inputs = [1, 2, 3, 2.5]

# Matrix (3,4)
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0,2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# Matrix (3,4)
weights = [[0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# Matrix (4,3) # Transpose -> dot product
# print(np.array(weights).T.shape) # Transpose 

output = np.dot(inputs, np.array(weights).T) + biases 
print(output)
'''