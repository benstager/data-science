import math


# Dot product
def dot_product(v,w):
    assert len(v) == len(w)
    return sum(i*j for i,j in zip(v,w))



# Subtraction
def vector_subtraction(v,w):
    assert len(v) == len(w) 
    return[i-j for i,j in zip(v,w)]

# Magnitude
def mag(v):
    return (math.sqrt(dot_product(v,v)))


# Distance function
def distance(v,w):
    assert len(v) == len(w)
    return (mag(vector_subtraction(v,w)))

x = [1,3,10]
y = [2,2,10]

print(vector_subtraction(x,y))
print(distance(x,y))