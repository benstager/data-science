import numpy as np
import matplotlib.pyplot as plt

# The goal of this code is to assign each data point to each centroid
# Also, to find the average of the each clustered data point to get a new centroid

def assign_centroids(X, centroids):
    # X - matrix m by n where each X[i] is a training point
    # centroids - k by n where each centroids[k] is a cluster centroid

    ids = np.zeros(X.shape[0])
    K = centroids.shape[0]

    # We iterate for each training set, and then find the
    # index of the one with the 2-norm squared minimized
    for i in range(X.shape[0]):
        arr = []
        for k in range(K):
            arr.append(np.linalg.norm(X[i] - centroids[k])**2)
        ids[i] = np.argmin(arr)
    
    return ids

def reassign_centroids(X, centroids, ids):
    # X - matrix m by n where each X[i] is a training point
    # centroids - k by n where each centroids[k] is a cluster centroid
    # ids - m by 1 array of which each training sets index assigned to each
    # centroid
    new_centroids = np.zeros(centroids.shape[0])
    K = centroids.shape[0]

    # Iterate over each centroid and find the points in matrix assosciated 
    # with that id. Then find the mean of each column and set the new centroid
    # index k with
    # np.mean calculates each features average
    for k in range(K):
        associates = X[ids == k]
        new_centroids[k] = np.mean(associates, axis = 0)
    
    return new_centroids