# for array computations and loading data
import numpy as np

# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# for building and training neural networks
import tensorflow as tf

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Let's try to select models using the cross validation/ test criteria

# Note that we can use a csv environment in numpy
data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')
X = data[:,0]
y = data[:,1]

# Change arrays to vectors that are treated as matrices
x = np.expand_dims(X, axis=1)
y = np.expand_dims(y, axis=1)


# We can split the data where X_, y_ is the 40% in the test set
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)

# Now split the remaining into two equal compartments.
# Note that the number argument is what the second arguments are segemented into
# First = 1 - test_size
X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size = .5, random_state = 1)
del X_, y_
scalar_linear = StandardScaler()

# Scale the data
X_train_scaled = scalar_linear.fit_transform(X_train)

# Write the lm
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Apply the model to the cross-validation data
X_cv_scaled = scalar_linear.fit_transform(X_cv)
yhat = linear_model.predict(X_cv_scaled)

# We'll only do this for this model, but calculate the MSE of each point
MSE = 0
for i in range(len(yhat)):
    MSE += (2/(len(yhat)))*(yhat[i] - y_cv)**2

print(MSE)

# Apply this to each model you use, choose lowest MSE, then report test MSE
# Money !