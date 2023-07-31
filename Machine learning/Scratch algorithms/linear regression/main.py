import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

"""
we want to write a linear regression algorithm from scrach
"""
# lets start by defining a function that computes a gradient of loss function
def compute_cost(X, y, w, b):
    m, n = X.shape
    cost = 0

    for i in range(m):
        cost += (1/2*m)*((np.dot(w,X[i]) + b) - y[i])**2

    return cost

def computes_gradient(X, y, w, b):
    dJ_dw = np.zeros(X.shape[0])
    db_dw = 0

    for i in range(X.shape[0]):
        dJ_dw += (1/X.shape[0])*X[i]*(np.dot(X[i],w) + b)
        db_dw += (1/X.shape[0])*(np.dot(X[i],w) + b)
    
    return dJ_dw, db_dw

# now we can define all of our parameters and hyperparameters and run gradient descent
X = np.random.random_sample([20,3])
y = np.random.random_sample([20,1])

alpha = .001
iters = 20
w = np.ones_like(X.shape[1])
b = 0
cost = []

def standardize(X):
    m, n = X.shape

    for j in range(n):
        X[:,j] = (X[:,j] - np.mean(X[:,j]))/np.std(X[:,j])

    return X

# now run gradient descent
def gradient_descent(X, y, w, b):
    
    for i in range(iters):
        dJ_dw, dJ_db = computes_gradient(X,y,w,b)
        w -= alpha*dJ_dw
        b -= alpha*dJ_db
    
    cost.append(compute_cost(X, y, w, b))
    
    return w, b, cost

plt.plot(range(iters), cost)
plt.show()
