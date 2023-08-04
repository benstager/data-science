import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
"""
k means scratch algorithm
"""


# 1. define k means function
def kmeans(X, k, iters):
    # generate random indices from 0-len(X), for k clusters
    idx = np.random.choice(len(X), k, replace=False)
    # returns a 3xn array of centroids
    centroids = X[idx,:]

    m, n = X.shape
    # disances will return an array of m x k, where each X[i] is an array of distance from point X[i] to each centroid
    distances = cdist(X, centroids, 'euclidean')
    # getting indices for each distance vector assigning an id
    indices = np.array([np.argmin(i) for i in distances])
    
    # iterating
    for _ in range(iters):
        centroids = []
        for idx in range(k):
            # calculating means of each data point assigned to each cluster and updating
            temp_centroid = X[indices == idx].mean(axis = 0)
            centroids.append(temp_centroid)
        
        centroids = np.vstack(centroids)
        distances = cdist(X, centroids, 'euclidean')
        indices = np.array([np.argmin(i) for i in distances])
    
    # return ids
    return indices
    


X = np.random.random_sample([10,3])
print(kmeans(X,3,10))