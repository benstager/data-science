import numpy 
import pandas
import matplotlib

from matplotlib import pyplot as plt

years = []
year = 1950
years.append(year)

for i in range(1,5):
    j = year + i*10
    years.append(j)

gdp = [300.2,543.3,1065,2862,10174]

# creating line chart

plt.plot(years,gdp,color = 'green', marker = 'o', linestyle = 'solid')
plt.title('test')
plt.xlabel('years')
plt.ylabel('gdp')
plt.show()

# creating bar chart

movies = ['a','b','c','d']
num_oscars = [4,2,3,10]

plt.bar(range(len(movies)),num_oscars)
plt.xticks(range(len(movies)),movies)
plt.show()


