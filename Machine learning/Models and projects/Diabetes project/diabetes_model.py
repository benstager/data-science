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
from sklearn.metrics import mean_squared_error
import math

"""
The goal of this project is to implement as LogisticRegression model using scikit
to determine a positive or negative case of diabetes. We will write
several models, first using numeric features only, and will as usual
pick the one with the lowest cross validation error.
"""

# first load data set using pandas
df = pd.read_csv('diabetes_prediction_dataset.csv')

# for the numerics, there is no cleaning to be done
features1 = ['HbA1c_level', 'blood_glucose_level']

X = df[features1]
y = df['diabetes']

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size= .3)

model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = model.predict(X_train)
print(mean_squared_error(y_hat, y_train))
print(1/7000*np.sum(np.square(y_train - y_hat)))