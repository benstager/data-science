from typing import Callable
import math
import random

def dot(u, v):
    assert len(u) == len(v)
    return sum([u_i*v_i for u_i, v_i in zip(u,v)])

def sum_of_squares(v):
    return dot(v,v)

# Estimating gradient

def difference_quotient(f, x, h):
    return ((f(x+h) - f(x)))/h
            
def partial_difference_quotient(f, X, i, h):
    X_h = [X_j + (h if j == i else 0) for j, X_j in enumerate(X)]
    return (f(X_h) - f(X))/h

def estimate_gradient(f,X,h):
    return [partial_difference_quotient(f, X, i, h) for i in range(len(X))]

# Estimating gradient in f(X) X in R^2

F = lambda X: X[0]**2 + X[1]**2 - 2*X[1]
X = [1,1]
h = .0001

def add(w,v):
    return [i + j for i, j in zip(w,v)]

def scalar_multiply(a, v):
    return [a*i for i in v]

def gradient_step(X, gradient, alpha):
    grad = scalar_multiply(-alpha, gradient)
    return add(X,grad)

X = [4,4]
h = 10**-7
alpha = .5


for epoch in range(1000):
    gradient = estimate_gradient(F, X, h)
    X = gradient_step(X, gradient, alpha)



    




