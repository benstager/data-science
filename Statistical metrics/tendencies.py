from collections import Counter

def mean(xs):
    return sum(xs)/len(xs)

def median(xs):
    sxs = sorted(xs)
    if len(sxs) % 2 == 0:
        midpoint = len(xs)//2
        return(((sxs[midpoint-1])+sxs[midpoint])/2)
    return sxs[len(sxs)//2]

assert median([1,10,2,9,5]) == 5
assert median([1,9,2,10]) == (2+9)/2
print('hi')