import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt

"""
we seek to write a logistic regression algorithm from scratch, by implementing
only numpy and pandas
"""

# 1. Standardization of data function
def standardization(X):
    m, n = X.shape

    for j in range(n):
        X[:,j] = (X[:,j] - np.mean(X[:,j]))/(np.std(X[:,j]))
    
    return X

# 2. sigmoid function
def sigmoid(x, w, b):
    sig = (1/(1+np.exp(-(np.dot(x, w)+b))))

    return sig


# 3. cost
def cost(X, y, w, b):
    cost = 0
    m, n = X.shape

    for i in range(m):
        cost += -(.5*m)*(y[i]*np.log2(sigmoid(X[i,:], w, b))  + (1-y[i])*np.log2(1 - sigmoid(X[i,:])))
    
    return cost

# 4. computing gradient
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dJ_dw = np.zeros(n)
    dJ_db = 0

    for i in range(m):
        dJ_dw += (1/m)*(sigmoid(X[i,:], w, b) - y[i])*X[i,:]
        dJ_db += (1/m)*(sigmoid(X[i,:], w, b) - y[i])
    
    return dJ_dw, dJ_db

# 5. define parameters
X = np.random.random_sample([10,4])
y = [1,1,0,1,0,0,0,1,0,1]
m, n = X.shape
w = np.ones_like(n)
b = 1
alpha = .001
iters = 20
cost = []

# 6. run gradient descent
def gradient_descent(X, y, w, b, iters, alpha):

    for i in range(iters):
        dJ_dw, dJ_db = compute_gradient(X, y, w, b)
        w -= alpha*dJ_dw
        b -= alpha*dJ_db
        cost.append(cost(X, y, w, b))

    return w, b, cost
x = range(iters)
plt.plot(x, cost)

# 7. making predictions
y_pred = sigmoid(X, w, b):
y_pred[y_pred >= .5] = 1
y_pred[y_pred < .5] = 0

# 8. various metrics
tp_tn = 0
accuracy = 0

for i in range(len(y)):
    if y[i] == y_pred[i]:
        tp_tn += 1

accuracy = (tp_tn)/len(y)

tp_fp = len(y_pred[y_pred == 1])
count1 = 0

for i in range(len(y)):
    if y_pred[i] == 1 and y[i] == 1:
        count1 += 1

precision = count1/tp_fp

tp_fn = 0

for i in range(len(y)):
    if y[i] == 1 and y_pred[0]:
        tp_fn += 1

recall = len(y[y == 1])/tp_fn
F1_score = (2*recall*precision)/(recall+precision)