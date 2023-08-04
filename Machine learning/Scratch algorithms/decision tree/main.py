import numpy as np
import pandas as pd 
import math
import random
import matplotlib.pyplot as plt
from collections import Counter

"""
we would like to manually code a decision tree for a given data set,
and we will use a class oriented structure to do so
"""

# entropy function for a series of categorical values
def entropy(y):
    hist = np.bincount(y)
    ps = hist/len(y)
    return -np.sum([p*np.log2(p) for p in ps if p > 0])

# writing node class
class Node:

    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def leaf_value(self):
        return self.value is not None
    
class DecisionTree:

    def __init__(self, min_samples_split = 2, max_depth = 100, n_feats = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = root

    def fit(self, X, y):
        self.feats = X.shape[1] if not self.feats else min(X.shape[1], self.n_feats)
        self.grow_tree(X, y)

    
