import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

# lab_utils_multi package isn't working, code wil not run

# 1. Load data
X, y = load_house_data()
features = ['size', 'bedrooms', 'floors', 'age']

# 2. Run descent
_, _, hist = run_gradient_descent()
plot_cost_i_w(X_train, y_train, hist)

# 3. Feature scale each column using Z-score normalization
# xi' = (x - xmean)/sigma
def zscore_normalization(X):

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu)/sigma

    return X_norm, mu, sigma

# 4. Rerun gradient descent and see if cost is decreasing
w_norm, b_norm, hist = run_gradient_descent(X, y, 1000, 1.0e-1)

# 5. Cost is now decreasing and we have gradient descent convergence
