import math
import numpy as np
import copy
from logistic_intro import sigmoid
from sklearn.linear_model import LogisticRegression
# We want to write a from scratch approach of gradient descent

# 1. Load data
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

# 2. Calculating partial derivatives of w and b
def compute_gradient(X, y, w, b):

    dj_dw = np.zeros(X.shape[1])
    dj_db = 0
    m, n = X.shape

    for i in range(m):
        for j in range(n):
            dj_dw[j] += (sigmoid(np.dot(X[i], w) + b) - y[i])*X[i,j]
        dj_db += sigmoid(np.dot(X[i], w) + b) - y[i]
    
    return (1/m)*dj_dw, (1/m)*dj_db

def descent(X, y, w_init, b_init, alpha, iter):

    w = w_init
    b = b_init

    for i in range(iter):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db

    return w, b

# 3. Initial guesses and parameters
w_tmp  = np.zeros_like(X[0])
b_tmp  = 0.
alpha = 0.1
iters = 10000

w, b = descent(X, y, w_tmp, b_tmp, alpha, iters)
print(np.matmul(X, w) + b)

# Let's compare using scikit
lr_model = LogisticRegression()
lr_model.fit(X, y)
print(lr_model.predict(X))

# Seems like a bug but whatever
