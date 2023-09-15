
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
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import torch

# read CSV and profile data
df = pd.read_csv('train.csv')

df.head(5)
df.info()
df.shape
df.describe()

fig = plt.figure(facecolor='white')

plt.pie(df['is_claim'].value_counts(), labels=['No Claim', 'Claim'], radius=1, colors=['green', 'orange'],
        autopct='%1.1f%%', explode=[0.1, 0.15], labeldistance=1.15, startangle=45,
        textprops={'fontsize': 15, 'fontweight': 'bold'})
plt.legend()
plt.show()

df.hist(figsize=(12,10))
plt.tight_layout
plt.show()
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

df_category = df[categories.keys()]

encoder = OrdinalEncoder()

encoder.fit(df_category)

df_category = encoder.transform(df_category)

df[categories.keys()] = df_category

"""
data_categories = df.select_dtypes(include=['object']).columns

df = pd.get_dummies(df, columns=data_categories,drop_first=True)
"""

"""
3. PCA, and visualizing first two principal components
"""

pca = PCA(n_components=2)

components = pca.fit_transform(df)

fig = px.scatter(components, x=0, y=1, color=df['is_claim'])

fig.show()

### data is highly oversampled in the negative class, need to resample
negative_class = df[df['is_claim']==0]
positive_class = df[df['is_claim']==1]

undersampled_majority = resample(
    negative_class,
    replace=False,
    n_samples=.5*len(positive_class),
    random_state=6
)

df_final = pd.concat([undersampled_majority,positive_class])

plt.pie(df_final['is_claim'].value_counts(), labels=['No Claim', 'Claim'], radius=1, colors=['green', 'orange'],
        autopct='%1.1f%%', explode=[0.1, 0.15], labeldistance=1.15, startangle=45,
        textprops={'fontsize': 15, 'fontweight': 'bold'})

plt.legend()
plt.show()
"""
4. fit model
"""

X = df_final.drop('is_claim',axis=1)
y_true = df_final['is_claim']

model = LogisticRegression()

model.fit(X, y_true)

y_predict = model.predict(X)

matrix = confusion_matrix(y_true, y_predict)

report = classification_report(y_true, y_predict)

# fitting model this way is poor, lets try 2 principal components

X = df_final['policy_tenure','age_of_policyholder']
y_true = df_final.drop('is_claim',axis=1)

model = LogisticRegression()

model.fit(X, y_true)

y_predict = model.predict(X)

matrix = confusion_matrix(y_true, y_predict)