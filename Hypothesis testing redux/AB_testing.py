import math
from typing import Tuple
from p_values import two_sided_p_value

def estimated_paramters(N, n):

    p = n/N
    sigma = math.sqrt(p*(1-p)/N)

    return p, sigma

def ab_test_stiatistic(N_a, n_a, N_b, n_b):

    p_A, sigma_A = estimated_paramters(N_a, n_a)
    p_B, sigma_B = estimated_paramters(N_b, n_b)

    return (p_B-p_A)/math.sqrt(sigma_A**2+sigma_B**2)

z= ab_test_stiatistic(1000, 200, 1000, 180)
print(two_sided_p_value(z))
