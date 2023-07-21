import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras import models
from keras import layers
from numpy import genfromtxt
import sklearn.metrics as sk_metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math
"""
We seek to write a number of models for the disease and symptom, and
outcome dataset. To begin we will write a number of logistic regression
models then choose the one with the least cross validation error.
"""


df = pd.read_csv('Disease_symptom_and_patient_profile_dataset.csv')

# Convert all categories to dummy variables
disease = pd.get_dummies(df['Disease'], drop_first = True)
fever = pd.get_dummies(df['Fever'], drop_first = True)
breathing = pd.get_dummies(df['Difficulty Breathing'], drop_first = True)
age = pd.get_dummies(df['Age'], drop_first = True)
gender = pd.get_dummies(df['Gender'], drop_first = True)
blood = pd.get_dummies(df['Blood Pressure'], drop_first = True)
cholest = pd.get_dummies(df['Cholesterol Level'], drop_first = True)

# Removing all labels
df.drop(['Disease','Fever',
           'Cough','Fatigue','Difficulty Breathing','Age','Gender','Blood Pressure',
           'Cholesterol Level'], axis = 1, inplace = True)

# Concatenating new matrix
df = pd.concat([df,disease, fever, breathing, age, gender, blood, cholest], axis = 1)

# Changing response
response = pd.DataFrame(df['Outcome Variable'])
response[response == 'Positive'] = 1
response[response == 'Negative'] = 0


df = df.drop(['Outcome Variable'], axis = 1)
df = df.apply(pd.to_numeric)
response = response.apply(pd.to_numeric)

# Writing logistic model
model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(df, response, 
                                                    test_size= .3, 
                                                    random_state= 0)

model.fit(X_train, y_train)

