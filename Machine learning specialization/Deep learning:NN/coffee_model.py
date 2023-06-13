import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# 1. Load the coffee data first
X, Y = load_coffee_data()

# 2. Scale the data
norm_l = tf.keras.layers.Normalization(axis = -1)
norm_l.adapt(X)
Xn = norm_l(X)

# 3. Writing neuro model
# We have 2 layers with 3 and 1 neurons, so set density that way
# Shape specifies the dimension of each training set excluding the intercept
model = Sequential(
    [
        tf.keras.Input(shape = (2,)),
        Dense(3, activation = 'sigmoid', name = 'layer1'),
        Dense(1, activation = 'sigmoid', name = 'layer2')
    ]
)

model.summary()

# Applying model execution to get weights to predict new data
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    X,Y,            
    epochs=10,
)

# New data to test
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test) # scale data
predictions = model.predict(X_testn) # Predict binary values for new data
print("predictions = \n", predictions)

# Prediction metric to test each new reaction point
y = np.zeros(predictions)
for i in predictions:
    if predictions[i] >= .5:
        y[i] = 'yes'
    else:
        y[i] = 'no'