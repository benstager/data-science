from typing import Tuple
import math

def normal(n,p):
    # n outcomes
    # p probability

    mu = p*n
    sigma = math.sqrt(p*(1-p)*n)
    return mu, sigma
