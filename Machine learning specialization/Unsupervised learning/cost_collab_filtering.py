import numpy as np

# We seek to manually write the regularized cost function for a collaborative
# filter
def cost_function(X, Y, W, b, R, lambda_):
    # X matrix of user features, size users by features
    # Y matrix of ratings, size movies by users
    # W matrix of user weights, size users by features
    # b 'matrix' of each intercept, size 1 by users
    # R matrix of 0s 1s for ratings, size movies by users
    # lambda_ regularizaton parameter

    J = 0
    # First calculate unregularized cost
    m, n = Y.shape

    for j in range(n):
        for i in range(m):
            J += np.square(R[i,j]*(np.dot(W[i,:], X[i,:]) + b[0,j] - Y[i,j]))
    
    J *= .5
    # Now proceed with regularization, we can use 
    # full vectorization of the entire matrix by summing the whole thing
    # Since there is no vector multiplication
    J += (np.sum(np.square(X))) + (np.sum(np.square(W)))
    J *= lambda_/2

    return J
