import matplotlib.pyplot as plt
import math 

sqrt2pi = math.sqrt(2*math.pi)

def normal_pdf(x,mu,sigma):
    return ((1/math.sqrt(2*math.pi)*sigma)*math.exp((-(x-mu)**2)/(2*sigma**2)))

xs = [x/10 for x in range(-50,50)]
print(xs)
# plt.plot(xs,[normal_pdf(x,2,1) for x in xs])
# plt.show()


def normal_cdf(x,mu,sigma):
    return (1+math.erf((x-mu)/math.sqrt(2)/sigma)) / 2

xs = [x/10 for x in range(-50,50)]
#plt.plot([normal_cdf(x,0,1) for x in xs])
#plt.show()

# Bisecting to find x with associated probability
def bisecting(target_p,x_low,x_high,tol):

    while x_high - x_low > tol:
        target_x = (x_high + x_low)/2
        mid_p = normal_cdf(target_x,0,1)
        if mid_p > target_p:
            x_high = target_x
        else:
            x_low = target_x
        
        target_x = (x_high + x_low)/2
        mid_p = normal_cdf(target_x,0,1)
    
    return target_x

print(bisecting(.6, -10, 10, .001))




