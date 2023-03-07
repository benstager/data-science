from collections import Counter
import random
from matplotlib import pyplot as plt
grades = []

# generating 10 random grades
for i in range(10):
    x = random.randint(0,100)
    grades.append(x)

# Counter function: returns dictionary of instances
# histogram is a dictionary in this case
histogram = Counter(min(grade//10*10,90) for grade in grades)

# Shifting bar over by 5 and plotting values
plt.bar([x+5 for x in histogram.keys()],histogram.values(),10,edgecolor = (0,0,0))
# axes
plt.axis([-5,105,0,5])
# x ticks from 0-100xs
plt.xticks([10*i for i in range(11)])
plt.show()

print(histogram)