import numpy as np 
import math
from logistic_intro import sigmoid

# Suppose we want to write a simple model of regularized LR

# 1. Load dummy data
X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y = np.array([460, 232, 178])

# 2. Write gradient computation function 

def compute_gradient(X, y, w, b, lambda_):

    m, n = X.size
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        for j in range(n):
            dj_dw[j] += (np.dot(X[i], w) + b - y[i])*X[i,j]
        dj_db += np.dot(X[i], w) + b - y[i]
    
    for j in range(n):
        dj_dw[j] += lambda_*w[j]
    
    return (1/m)*dj_dw, (1/m)*dj_db

# 3. Running gradient descent leads to the same thing, same with logistic gradient