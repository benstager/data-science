
"""
the goal of this exercise is to implement a logistic regression
model from scikit-learn to predict if a customer will file an
insurance claim in the next 6 months
"""

"""
1. package management, data reading and profiling
"""

# importing necessary libraries
import numpy as np
import pandas as pd
import math
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import torch

# read CSV and profile data
df = pd.read_csv('train.csv')

df.head(5)
df.info()
df.shape
df.describe()

"""
2. data cleaning and preprocessing
"""

df.isna()

df = df.dropna(axis=0)

data_types = dict(enumerate(df.dtypes.values))

category_indices = [i for i in data_types.keys() if data_types[i] == 
       'object']

df = df.drop(['policy_id', 'max_torque','max_power','engine_type']
             ,axis=1)

categories = df.select_dtypes(include='object')

categories.info()

encoder = OneHotEncoder()

encoder.fit(categories)

categories = encoder.transform(categories)

# still needs to be concatenated with OG data frame

