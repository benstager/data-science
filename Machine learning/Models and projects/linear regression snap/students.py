import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

"""
simple linear regression model between hours studies and scores
"""

# load csv
df = pd.read_csv('hoursScores.csv')
df.isnull().sum() # searching for null or empty columns

sns.scatterplot(data=df, x='Hours', y='Scores')
plt.show()

sns.regplot(data=df, x='Hours', y='Scores')
plt.show()

# now create features
X = df[['Hours']]
y = df['Scores']

# train test split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2, random_state= 101)

model.fit(X_train, y_train)
yhat_test = model.predict(X_test)

# now lets evaluate our model
error = mean_squared_error(y_test, yhat_test)

# model summary
print(model.score(X_train,y_train))

