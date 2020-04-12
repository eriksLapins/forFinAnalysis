# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:36:24 2020

@author: User
"""

import os
os.getcwd()
os.chdir("C:/Users/User/Investment/W_Report/Python/Projects/forFinAnalysis")
#%%
# import packages for analysis

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#%%

# import data as a data frame

fb = pd.read_csv('facebook.csv', index_col='Date')
fb.head()
ms = pd.read_csv('microsoft.csv', index_col='Date')
ms.head()
# show the type of data

print(type(fb)) # .DataFrame shows that it is a data frame

# give first 5 rows of dataframe

fb.head()

fb.index
fb.index[0] # the first index
fb.index[-1] # the last index
fb.columns
fb.shape # number of rows and columns

fb.tail() # last 5 rows
fb.tail(10) # last 10 rows

fb.describe()

# slicing data frame

fb.loc['2015-01-02', 'Close'] # slice by number of index
                                # show the cell in this location [<index>, <column>]

fb.iloc[1, 3] # slice by number of rows
                # show the cell in row 1, column 3 (positions start with 0; thus,
                    # column 1 is actually column 2)

fb.loc['2015-01-01':'2015-12-31', 'Close'] # get the close price for the whole year 2015

# plotting without matplotlib

fb.loc['2015-01-01':'2015-12-31', 'Close'].plot()

# select one column

fb['Close']

# select two columns

fb[['Open', 'Close']]

# Create a new column

fb['Price1'] = fb['Close'].shift(-1) # tomorrow's stock close price
            # .shitf(-1) - shifts the column upwards by one row

# Price Difference = (close price tomrrow - close price today)

fb['PriceDiff'] = fb['Price1'] - fb['Close']
fb.head()

# Return calculation

fb['Return'] = fb['PriceDiff']/fb['Close']

# Showing the direction of price changes

    # List comprehension

fb['Direction'] = [1 if fb.loc[ei,'PriceDiff'] > 0 else -1
                   for ei in fb.index] # ei is any index of fb. Direction will be 
                                        # valued according to the condition defined 
                                        # in the second line of this code (starting with for)
print('Price difference on {} is {}. direction is {}'.format('2015-01-05', # set the {} to display the specific value
                                fb['PriceDiff'].loc['2015-01-05'], # second {} is price difference
                                fb['Direction'].loc['2015-01-05'])) # third {} is direction

#%%
# Creating a "moving average"

fb['Average3'] = (fb['Close'] + fb['Close'].shift(1)+fb['Close'].shift(2))/3 
                                        # Three day moving average
# shift 1 shifts a column one row down
# shift 2 shifts a column down by two rows

# Calculate moving average using .rolling()

fb['MA40'] = fb['Close'].rolling(40).mean() # 40 day moving average
fb['MA200'] = fb['Close'].rolling(200).mean() # 200 day moving average

# Plot with data + moving averages
plt.figure(figsize=(10, 8))
fb['Close'].plot()
fb['MA40'].plot() # Fast signal
fb['MA200'].plot() # Slow signal
plt.legend()
plt.show()

#%%
# plotting moving averages without having to calculate as much

fb['ma50'] = fb['Close'].rolling(50).mean()

#plot the moving average
plt.figure(figsize=(10, 8))
fb['ma50'].loc['2015-01-01':'2015-12-31'].plot(label='MA50')
fb['Close'].loc['2015-01-01':'2015-12-31'].plot(label='Close')
plt.legend()
plt.show()

#%%
# Simple trading strategy







