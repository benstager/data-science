import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
writing a logistic regression binary classifier using a mixture of several features
"""

# create a dataframe 
df = pd.read_csv('framingham.csv')
df.info()
df.isnull().sum()

# dropping education
df = df.drop('education', axis='columns')

# rid dataframe of null rows
df = df.dropna()
print(df.shape)

# data is now cleaned, lets establish data
y = df[['TenYearCHD']]
X = df.drop('TenYearCHD', axis=1)
print(X.shape)
print(df.shape)

# lets scale the data using StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X = scaler.fit_transform(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

# model fitting
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
yhat = model.predict(X_test)
print(yhat)

# evaluation metrics
from sklearn.metrics import confusion_matrix, accuracy_score,  classification_report

confusion = confusion_matrix(y_test, yhat)

test_acc_score = accuracy_score(y_test, yhat)
print(test_acc_score)

test_class_report = classification_report(y_test, yhat)
print(test_class_report)