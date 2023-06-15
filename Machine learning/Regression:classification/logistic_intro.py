import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):

    return 1/(1 + np.exp(-x))

# Test sigmoid function
X = np.arange(-5,6)
y = sigmoid(X)

