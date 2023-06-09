import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# We have binary features. X[i,:] is one data point
# Note that each member of X[i,:] is true or false for each feature
X = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

# 1 -> cat, 0 -> not cat
y = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

# Writing a manual entropy function
# p is fraction of true/current node
def entropy(p):

    if p == 0 or p == 1:
        return 0
    else:
        return -p*np.log2(p) - (1 - p)*np.log2(1 - p)
    
# Given a feature, left node == 1, right node == 0
# index_feature is the feature that will be split
def indices_split(X, index_feature):

    left_indices = []
    right_indices = []

    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    
    return left_indices, right_indices

# We iterate over X's each data point and specific indice to test
# The indices return are the data points with that specific feature
# Let's now calculate the associated entropy of each left and right node

left, right = indices_split(X, 0)

def calcualate_entropy(X, y, left_indices, right_indices):
    
    ent = 0

    w_left = len(left_indices)/len(y)
    w_right = len(left_indices)/len(y)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    
    ent = (w_left*entropy(p_left) + w_right*entropy(p_right))
    return ent

def information_gain(X, y, left_indices, right_indices):

    # Probability of root node
    p_node = sum(y)/len(y)
    ent = calcualate_entropy(X, y, left_indices, right_indices)

    return entropy(p_node) - ent

# Now iterating over each feature
for i, feature_name in enumerate(['ears', 'face', 'whiskers']):
    left_indices, right_indices = indices_split(X, i)
    i_gain = information_gain(X, y, left_indices, right_indices)
    print(i_gain)



