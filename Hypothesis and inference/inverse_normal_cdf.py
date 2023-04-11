import normal_cdf
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