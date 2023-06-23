import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras import models
from keras import layers
from numpy import genfromtxt
from keras import Sequential
from keras.layers import Dense
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import math

"""
The goal of this project is to model various hepatitis data to predict
gender outcome. We will implement several features using 
simple logistic regression then implement a neural network to see
if the cross-validation error is less

"""

# 1. Load data and identify features
col_names = ['ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT']
df = pd.read_csv("HepatitisCdata.csv")

# we have some NA rows so we need to clean a bit
indices = df[(df['ALB'] == 1000) | (df['ALP'] == 1000) | (df['ALT'] == 1000) | (df['AST'] == 1000) | (df['BIL'] == 1000) | (df['CHE'] == 1000) | (df['CHOL'] == 1000) | (df['CREA'] == 1000) | (df['GGT'] == 1000) | (df['PROT'] == 1000)].index
df.drop(indices, inplace = True)

# done cleaning, lets establish model data
X = df[col_names]
y = df['PROT']

# now writing simple model with all features, first split data to be used
# in all models, training size is 70% of trainings ets
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = .3)
cv = []
model = LinearRegression()

# MODEL 1: all features, predict X_cv and find CV MSE
model.fit(X_train, y_train)
y_hat1 = model.predict(X_cv)
cv.append(np.sum(np.square(y_hat1 - y_cv)))

# MODEL 2: one feature
X = np.array(df['ALB'])
X = X.reshape(-1,1)
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = .3)

model.fit(X_train, y_train)
y_hat2 = model.predict(X_cv)
cv.append(np.sum(np.square(y_hat2 - y_cv)))

# MODEL 3: using tensorflow NN
"""
X = df[col_names]
y = df['PROT']
X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = .3)

model = Sequential([
    Dense(14, 'linear'),
    Dense(7, 'linear'),
    Dense(1, 'linear')
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = .01) 
)

model.fit(X_train, y_train)
y_hat3 = np.array(model.predict(X_cv))
y_hat3 = y_hat3.reshape(-1,1)
"""

"""
The final goal for this model, apart from testing some ML frameworks is to
determine if a NN is superior to that of a simple Logistic Regression
model for determining gender of different hepatitis patients. 
NEED TO FIX UNKNOWN TYPE ERROR FOR LOGISTIC REGRESSION
"""

# lets first reload the data and see what we can do
X = df[col_names]
y = np.array(df['Sex'])

y[y == 'm'] = 1
y[y == 'f'] = 0
y=y.astype('int')

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size= .3)

model = LogisticRegression()
model.fit(X_train, y_train)
print(model.predict(X_train))

# Logistic model is done !