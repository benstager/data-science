import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('severance_pay.csv')
df.info()
df.head(5)