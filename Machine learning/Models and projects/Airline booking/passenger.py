"""
Taking in data about passengers taking various flights to make certain predictions
"""

"""
1. Establish initial modules and dataframes
"""

import numpy as np
import pandas as pd

import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

df = pd.read_csv('Passanger_booking_data.csv')
df_final = df  #store original dataframe



"""
One-hot encode the categorical variables
"""

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown= 'ignore')

"""
One-hot encode sales channel and trip type, two categorical variables
"""

encoder_df = pd.DataFrame(encoder.fit_transform(df[['sales_channel']]).toarray())
encoder_df = encoder_df.rename(columns = {0: 'Internet', 1: 'Mobile'})
df_final = df_final.join(encoder_df)

encoder_df = pd.DataFrame(encoder.fit_transform(df[['trip_type']]).toarray())
encoder_df = encoder_df.rename(columns = {0: 'RoundTrip', 1: 'OneWayTrip', 2: 'CircleTrip'})
df_final = df_final.join(encoder_df)

df_final.drop(['sales_channel', 'trip_type', 'route', 'booking_origin'], axis = 1, inplace=True) # not going to use other categorical variables

y = df_final['booking_complete']
df_final.drop('booking_complete', axis=1, inplace=True) # inplace augments original dataframe

print(df_final.head(5))
