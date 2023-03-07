import matplotlib.pyplot as plt
import random

def uniform_pdf(x):
    return 1 if 0 <= x <= 1 else 0

x = []
for i in range(100):
    j = random.random() 
    x.append(uniform_pdf(j))

plt.plot(x)
plt.show()