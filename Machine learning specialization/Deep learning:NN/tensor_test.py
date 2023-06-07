import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Activation
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.activations import sigmoid, deserialize
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# 1. Setting arbitrary data
X = np.array([[1.0], [2.0]], dtype=np.float32)           
y = np.array([[300.0], [500.0]], dtype=np.float32) 

# 2. Using tensorflow to set a layer
#    Units = 1 assumes 1 neuron?
linear_layer =  tf.keras.layers.Dense(units = 1)
linear_layer.get_weights()

# Reshape vector so it is a column
a1 = linear_layer(X[0].reshape(1,1))
w, b = linear_layer.get_weights()

set_w = np.array([[200]])
set_b = np.array([100])
linear_layer.set_weights([set_w, set_b])

a1 = linear_layer(X[0].reshape(1,1))
print(a1)
alin = np.dot(set_w,X[0].reshape(1,1)) + set_b
print(alin)