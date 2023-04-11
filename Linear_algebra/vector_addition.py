from typing import List

# Creates vector of each entry type float

Vector = List[float] 

# Crude way of vector addition
def add_crude(v,w):
    x = []
    # debugging statement
    assert len(v) == len(w)
    for i,j in zip(v,w):
        x.append(i+j)
    
    return x
    # return [vx + wx for vx, wx in zip(v,w)]

# Effecient vector addition
def add_new(v,w):
    assert len(v) == len(w)
    return [i + j for i,j in zip(v,w)]


print('test1',add_crude([1,23,3],[4,5,6]))
print('test2',add_new([1,23,3],[4,5,6]))
