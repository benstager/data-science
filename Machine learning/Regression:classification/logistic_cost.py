import numpy as np
from logistic_intro import sigmoid
import math

# 1. Load random data
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y = np.array([0, 0, 0, 1, 1, 1])   

# 2. Define logistic cost function based on sigmoid
def logistic_cost_test(X, y, w, b):

    m = X.shape[0]
    cost = 0

    for i in range(m):
        cost += -y[i]*math.log(sigmoid(np.dot(X[i], w) + b)) - (1 - y[i])*math.log(1 - sigmoid(np.dot(X[i], w) + b))
    
    return (1/m) * cost

# 3. Unit test
w = np.array([1,1])
b = -3
print(logistic_cost_test(X, y, w, b))
x, y = X.shape
print(x,y)