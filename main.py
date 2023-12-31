

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

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # [0,1] or [[0,1], [1,0]] (one hot encode)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


 
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

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss: ", loss)


'''
# derivatives


import matplotlib.pyplot as plt
import numpy as np 

def f(x):
    return 2*x**2


x = np.arange(0, 50, 0.001)
y = f(x)

#print(x)
#print(y)

plt.plot(x,y)
#plt.show()

colors = ['k', 'g', 'r', 'b', 'c']


def approximate_tangent_line(x, approximate_derivative, b):
    return approximate_derivative*x + b

for i in range(5):
    p2_delta = 0.0001

    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    print((x1,y1),(x2,y2))

    approximate_derivative = (y2-y1) / (x2-x1)
    b = y2 - approximate_derivative*x2




    to_plot = [x1-0.9, x1, x1+0.9]

    plt.scatter(x1, y1, c=colors[i])
    plt.plot(to_plot, 
            [approximate_tangent_line(point, approximate_derivative, b) 
                for point in to_plot], colors[i])


    print('Approximate derivative for f(x)', f'where x= {x1} is {approximate_derivative}')

plt.show()



'''
