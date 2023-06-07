import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Writing a simple implementation of a scratch a out layer calculator
g = sigmoid
def my_dense(a_in, W, b):
    
    units = W.shape[1]
    a_out = np.zeros(units)
    m = len(a_in)
    
    for i in range(units):
        a_out[i] = g(np.dot(a_in, W[i]) + b)
    
    return a_out