import numpy as np
import math
import matplotlib.pyplot as plt
import random
from collections import Counter

"""
we seek to implement KNN from scratch
"""

# 1. Define all necessary functions
def knn(data, query, k, distance_fn, choice_fn):
    
    neighbors_distances_indices = []

    for index, example in enumerate(data):
        distance = distance_fn(example[:-1], query)
        neighbors_distances_indices.append((distance,index))

    sorted_list = sorted(neighbors_distances_indices)
    sorted_k = sorted_list[:k]
    sorted_labels = [data[i][-1] for distance, i in sorted_k]

    return sorted_k, choice_fn(sorted_labels)

def mean(y):
    return sum(y)/len(y)


def mode(y):
    return Counter(y).most_common(1)[0][0]

def euclidean_distance(x1, x2):
    return np.linalg.norm(np.subtract(x1,x2))


reg_data = [
       [65.75, 112.99],
       [71.52, 136.49],
       [69.40, 153.03],
       [68.22, 142.34],
       [67.79, 144.30],
       [68.70, 123.30],
       [69.80, 141.49],
       [70.01, 136.46],
       [67.90, 112.37],
       [66.49, 127.45],
    ]
query = [60]

reg_k_nearest_neighbors, labels = knn(reg_data, query, 3,
                                              distance_fn=euclidean_distance, choice_fn= mean)

print(labels)
