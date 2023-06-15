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
    step_size = (max(p_val) - min(p_val))/1000

    for epsilon in np.arange(min(p_val), max(p_val), 1000):

        predictions = (p_val < epsilon)

        # true positives: predicted flag, true flag
        tp = sum((predictions == 1) & (y_val == 1))
        # false positives: predicted flag, no true flag
        fp = sum((predictions == 1) & (y_val == 0))
        # false negatives: predicted no flag, true flag
        fn = sum((predictions == 0) & (y_val == 1))

        prec = tp/(tp + fp)
        rec = tp/(tp + fp)
        F1 = (2*prec*rec)/(prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    
    return best_F1, best_epsilon