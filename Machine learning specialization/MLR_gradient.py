import copy, math
import numpy as np
import matplotlib.pyplot as plt

# We want to find coeffcients for gradient descent for MLR

# 1. Data
X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y = np.array([460, 232, 178])
# Each X[i] is an data point

# 2. Initial guesses of weights beta
beta_0 = 785.1811367994083
beta_ls = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

# 3. Computing gradient for each iteration
def gradient_compute(X, y, beta_0, beta_ls):

    m, n = X.shape
    grad_beta0 = 0
    grad_betals = np.zeros(n)

    for i in range(m):
        for j in range(n):
            grad_betals[j] += (np.dot(X[i], beta_ls) - y[i])*X[i,j]
        grad_beta0 += (np.dot(X[i], beta_ls) - y[i])
    
    return 1/m*grad_beta0, 1/m*grad_betals

# 4. Descent 
def gradient_descent(X, y, beta_0init, beta_lsinit, alpha, iter):

    b = beta_0init
    w = beta_lsinit

    for i in range(iter):
        beta_0, beta_ls = gradient_compute(X, y, b, w)
        b = b - alpha*beta_0
        w = w - alpha*beta_ls

    return b, w

# 5. Test
alpha = 5.0e-7
iter = 1000
initial_w = np.zeros(4)
initial_b = 0

print(gradient_descent(X, y, initial_b, initial_w, alpha, iter))
        
    

