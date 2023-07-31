import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

"""
we seek to make predictions about if customer will make a deposit using a RandomForest classifier
and a DecisionTree classifier
"""

# first, load in data
df = pd.read_csv('banking.csv')

# check for any null rows
df.isnull().sum()

# we have null rows so we can use dropna to get rid of any data points will null values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN
df.dropna(inplace=True)
print(df)

# lets visualize some of the data
plt.subplot(1,2,2)
sns.countplot(x='y', data=df)

# seek categorical variables
categories = [cols for cols in df.columns if df[cols].dtype == 'object']
print(categories)

# we have a lot of categorical data so we must convert this to discrete values
encoder = LabelEncoder()

df['job'] = encoder.fit_transform(df['job'])
df['marital'] = encoder.fit_transform(df['marital'])
df['education'] = encoder.fit_transform(df['education'])
df['job'] = encoder.fit_transform(df['job'])
df['default'] = encoder.fit_transform(df['default'])
df['loan'] = encoder.fit_transform(df['loan'])
df['contact'] = encoder.fit_transform(df['contact'])
df['poutcome'] = encoder.fit_transform(df['poutcome'])
df['housing'] = encoder.fit_transform(df['housing'])

# manually encoding months to match calendar value
df['month'] = df['month'].map({'jan':1,
                               'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                               'jul':7,'aug':8,'sept':9,'oct':10,'nov':11,'dec':12})

print(df.head(5))

# start establishing features and labels
X = df.drop('y', axis=1)
y = df['y']

# scale only id and age
scaler = MinMaxScaler(copy=True, feature_range=(0,1))
X = scaler.fit_transform(X)

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=22)

# writing model in RandomForest


rf = RandomForestClassifier(criterion= 'gini', n_estimators=100, max_depth=3, random_state=22)
rf.fit(X_train, y_train)

y_pred1 = rf.predict(X_test)

print(rf.score(X_train,y_train))
print(rf.score(X_test, y_test))
print(classification_report(y_test, y_pred1))

