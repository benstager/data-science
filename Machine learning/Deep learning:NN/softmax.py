import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models, layers
from keras.models import Sequential
from keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# 1. We can manually denote the probability of P(y = n) as:

def my_softmax(z):
    return np.exp(z)/np.sum(z)

# 2. Generate some random data
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X, y = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

# 3. Write softmax model with 4 probabilities

model = Sequential(
    [
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')
    ]
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(.001)
)

# Fitting to data
model.fit(
    X,y,
    epochs = 10
)


# Getting each probability for each data point
p_nonpreferred = model.predict(X)