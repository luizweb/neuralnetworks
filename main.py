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

# parei em 11:45    