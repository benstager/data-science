import numpy as np
import matplotlib.pyplot as plt

# Function returns metrics mean and standard deviation given 
# X data matrix
def mean_std(X):
    m, n = X.shape
    mu = 1/m * np.sum(X, axis = 0)
    sigma = 1/m * np.sum((X-mu)**2, axis = 0)

    return mu, sigma

def select_threshold(y_val, p_val):
    # p_vals are probabilities of each X[i] in X
    # y_vals are the ground truth (1 bad 0 good) on data set

    best_epsilon = 0
    best_F1 = 0

    for epsilon in np.arange()