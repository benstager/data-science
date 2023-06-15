import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2])
y_hat = np.array([5, 6])

# number of data points

n = x.shape[0]
n = len(x)

plt.scatter(x, y_hat, marker = 'x', c = 'r')
plt.title('Test')
plt.xlabel('x')
plt.ylabel('y')
plt.show()