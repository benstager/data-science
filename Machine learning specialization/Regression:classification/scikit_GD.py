import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

# We want to write a model using scikit and GD

# 1. Load data even though package isn't work
X, y = load_house_data()
features = ['size', 'bedrooms', 'floors', 'age']

# 2. Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Creating model using GD
sgdr = SGDRegressor(max_iter = 1000)
sgdr.fit(X, y)
print(sgdr)

# 4. Seeing parameters
beta_0 = sgdr.intercept_
beta_ls = sgdr.coef_

# 5. Predicting current design matrix

y_pred = sgdr.predict(X)
