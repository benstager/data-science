import normal_cdf
import inverse_normal_cdf
import normal
from typing import Tuple
import math

def inverse_normal_cdf(p,mu,sigma,tol):
    
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tol = tol)

    low_z = -10
    high_z = 10

    while high_z - low_z > tol:
        mid_z = (low_z+high_z)/2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            high_z = mid_z
    
    return mid_z

def normal_probability_above(lo,mu,sigma):
    return (1 - normal_cdf(lo,mu,sigma))

def normal(n,p):
    # n outcomes
    # p probability

    mu = p*n
    sigma = math.sqrt(p*(1-p)*n)
    return mu, sigma

def normal_probability_range(lo,hi,mu,sigma):
    return (normal_cdf(hi,mu,sigma) - normal_cdf(hi,mu,sigma))

def normal_probability_low(lo,mu,sigma):
    return (normal_cdf(lo,mu,sigma))

def normal_upper_bound(probability,mu,sigma):
    return inverse_normal_cdf(probability,mu,sigma,.01)

def normal_lower_bound(probability,mu,sigma):
    return inverse_normal_cdf(1-probability,mu,sigma)

def normal_two_sided_bounds(probability,mu,sigma):
    # returns outcome bounds that generate probability

    tail_probability = (1-probability)/2

    upper_bound = normal_lower_bound(tail_probability,mu,sigma)

    lower_bound = normal_upper_bound(tail_probability,mu,sigma)

    return lower_bound, upper_bound


mu_0, sigma_0 = normal(1000,.5)

lower_bound, upper_bound = normal_two_sided_bounds(.5, mu_0, sigma_0)
