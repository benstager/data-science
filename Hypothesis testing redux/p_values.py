from cdfs_pdfs import normal_probability_below, normal_probability_above,normal_approximation_to_binomial

def two_sided_p_value(x, mu = 0, sigma = 1):

    if x >= mu:
        return 2*normal_probability_above(x, mu, sigma)
    else:
        return 2*normal_probability_below(x, mu, sigma)


mu_0, sigma_0 = normal_approximation_to_binomial(1000, .5)
