import numpy as np
import matplotlib.pyplot as plt
import math

x_train = np.array([1.0, 2.0])    
y_train = np.array([300.0, 500.0]) 

def cost(x, y, beta0, beta1):

    RSS = 0
    for i in range(len(x)):
        RSS += ((beta0 + beta1*x[i]) - y[i])**2
   
    return 1/len(x) * RSS

print(cost(x_train, y_train, 200, 5))