from typing import Callable

E = [[1,2,3],
    [4,5,6],
    [7,8,9]]

# Returns matrix dimensions
def shape(A):
    return len(A), len(A[0]) 

# Matrix creator
# Returns entry function for m rows and n columns
def create(m,n,entry_fn):
    return [[entry_fn(i,j) for i in range(m)] for j in range(n)]

# Identity creator

# lambda function implemented
# The function returns a 1 if the indices are equal, otherwise returns 0
# lambda function is similar to MATLAB anonymous function
# lambda arguments: expression
def identity(n):
    return create(n,n, lambda i, j: 1 if i == j else 0)

print(identity(5))

# sample lambda function similar to matlab implementation
f = lambda x: x*x

