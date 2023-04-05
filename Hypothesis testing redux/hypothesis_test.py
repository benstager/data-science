from cdfs_pdfs import normal_cdf, inverse_normal_cdf, normal_approximation_to_binomial
from typing import Tuple

def normal_upper_bound(probability, mu = 0, sigma = 1):
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability, mu = 0, sigma = 1):
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu = 0, sigma = 1):
    
    tail_probability = (1 - probability)/2

    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, .5)
lower_bound, upper_bound = normal_two_sided_bounds(.95, mu_0, sigma_0)
print(lower_bound, upper_bound)