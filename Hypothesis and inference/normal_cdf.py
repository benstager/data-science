import math

def normal_cdf(x,mu,sigma):
    return (1+math.erf((x-mu)/math.sqrt(2))/2)

