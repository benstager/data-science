import math
from typing import Tuple
def normal_cdf(x, mu = 0, sigma = 1):
    return (1+math.erf((x-mu)/math.sqrt(2)/sigma))/2

def inverse_normal_cdf(p, mu = 0, sigma = 1, tol = .01):

    if mu != 0 or sigma != 1:
        return mu + sigma*inverse_normal_cdf(p)
    
    low_z = -10
    high_z = -10 
    mid_z = (low_z + high_z)/2
    while high_z - low_z > tol:
        mid_z = (low_z + high_z)/2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            high_z = mid_z
    
    return mid_z

def normal_upper_bound(probability, mu = 0, sigma = 1):
    return inverse_normal_cdf(probability, mu, sigma)

def normal_probability_above(lo, mu = 0, sigma = 1):
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_below(lo, mu = 0, sigma = 1):
    return normal_cdf(lo, mu, sigma)

def normal_probability_between(lo, hi, mu = 0, sigma = 1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_approximation_to_binomial(n, p):
    
    mu = p*n
    sigma =  math.sqrt(p*(1-p)*n)

    return mu, sigma 

