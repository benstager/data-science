from scipy.stats import geom
import matplotlib.pyplot as plt
import numpy as np

#creating an array of values between
#1 to 20 with a difference of 1
x = np.arange(1, 20, 1)
   
y = geom.pmf(x, 0.25)
   
plt.plot(x, y, 'bo') 
plt.show()