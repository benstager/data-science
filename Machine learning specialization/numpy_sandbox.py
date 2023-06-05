import numpy as np
import time

x = np.random.rand(50)
y = np.random.rand(50)

# Let's do a simple test to compare comp time between for and vectorization

# 1: for loop
z = 0

tic = time.time()
for i in range(len(x)):
    z += x[i]*y[i]
toc = time.time()

# 2: vectorization

tic = time.time()
z = np.dot(x,y)
toc = time.time()



### Practicing quickly with matrices should follow similarly to MATLAB
A = np.random.random_sample((2,2))
print(A)
print(A[:,0])
print(A[0,:])