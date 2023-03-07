import random
from collections import Counter
import matplotlib.pyplot as plt
import math


def normal_pdf(x,mu,sigma):
    return ((1/math.sqrt(2*math.pi)*sigma)*math.exp((-(x-mu)**2)/(2*sigma**2)))

def normal_cdf(x,mu,sigma):
    return (1+math.erf((x-mu)/math.sqrt(2)/sigma)) / 2

def bernoulli_trial(p):
    return 1 if random.random() < p else 0

def binomial(n,p):
    return sum(bernoulli_trial(p) for i in range(n))


print(binomial(20,.25))

def binomial_histogram(n,p,num_points):

    data = [binomial(n,p) for i in range(num_points)]
    histogram = Counter(data)
    plt.bar([x for x in histogram.keys()], [y/num_points for y in histogram.values()])

    mu = n*p
    sigma = math.sqrt(n*p*(1-p))
    xs = range(min(data),max(data)+1)
    ys = [normal_cdf(i+.5, mu, sigma) - normal_cdf(i-.5,mu,sigma) for i in xs]
    plt.plot(xs,ys)
    plt.show()






binomial_histogram(100,.75,10000)
print(normal_cdf(90, 100*.5, math.sqrt(100*.5*(1-.5)))-normal_cdf(80, 100*.75, math.sqrt(100*.5*(1-.5))))