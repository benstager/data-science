import math, copy
import numpy as np
import matplotlib.pyplot as plt
from cost_example import cost

xs = np.array([1, 2])
ys = np.array([300, 500])


# Gradient function only works for cost example
# Iterates over all x,y pairs to return relative gradient using RSS
def gradient(x, y, beta0, beta1):

    dbeta0 = 0
    dbeta1 = 0
    n = len(x)

    for i in range(n):
        dbeta0 += (beta0 + beta1*x[i])
        dbeta1 += (beta0 +beta1*x[i])*x[i]
    
    return 1/n * dbeta0, 1/n * dbeta1

def gradient_descent(x, y, alpha, beta0, beta1, iter):

    for i in range(iter):
         d0, d1 = gradient(x, y, beta0, beta1)
         beta0 = beta0 - alpha*d0
         beta1 = beta1 - alpha*d1

    return beta0, beta1

print(gradient_descent(xs, ys, .5, .5, .5, 10**4))

