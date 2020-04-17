# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:31:04 2020

@author: User
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as norm

#%%
import os
os.getcwd()
os.chdir("C:/Users/User/Investment/W_Report/Python/Projects/forFinAnalysis/week4")
#%%
housing = pd.read_csv('housing.csv', index_col=0)
housing = housing[['medv','rm','nox', 'indus', 'lstat']]
housing.head()
#%%
housing.cov() # covariances of all variables

housing.corr() # correlations of all variables

#%%
#scatterplot
from pandas.plotting import scatter_matrix
sm = scatter_matrix(housing, figsize=(10,10))

#%%
# This time we take a closer look at MEDV vs LSTAT。 What is the association between MEDV and LSTAT you observed?
housing.plot(kind='scatter', x='lstat', y='medv', figsize=(10, 10))

#%% 
# simple linear regression

b0 = 1
b1 = 2
housing['GuessResponse'] = b0 + b1*housing['rm']

housing['observederror']=housing['medv']-housing['GuessResponse']
indices = [7,20,100]
print(housing['observederror'].loc[indices])

print('Sum of squared errors is ', (housing['observederror']**2).sum())

#%%
# regression less manually
import statsmodels.formula.api as smf
model = smf.ols(formula = 'medv~rm', data=housing).fit()

b0 = model.params[0]
b1 = model.params[1]
housing['BestResponse'] = b0+b1*housing['rm']

#%% plotting the best fit line

plt.figure(figsize=(10,10))
plt.scatter(housing['rm'], housing['medv'], color='g', label='real')
plt.scatter(housing['rm'], housing['GuessResponse'], color='red')
plt.scatter(housing['rm'], housing['BestResponse'], color='yellow')
plt.ylabel('medv/$1000')
plt.xlabel('rm/number')
plt.xlim(np.min(housing['rm'])-2, np.max(housing['rm'])+2)
plt.ylim(np.min(housing['medv'])-2, np.max(housing['medv'])+2)
plt.legend()
plt.show()
#%%

# statistical evaluation of our model (summary)
model.summary()

#%%
# residual plot
housing['error'] = housing['medv'] - housing['BestResponse']
plt.figure(figsize=(15,8))
plt.title('Residuals vs order')
plt.plot(housing.index, housing['error'], color='purple')
plt.axhline(y=0,color='red')
plt.show()

# Durbin watson test for serial correlation - rule of thumb - 1.5-2.5 is normal
            # below 1.5 - positively correlated, if above 2.5 - negatively correlated
            # if 1.5-2.5 assumptions are not violated
#%%
import scipy.stats as stats
import matplotlib.pyplot as plt
z = (housing['error'] - housing['error'].mean())/housing['error'].std(ddof=1)

stats.probplot(z, dist='norm', plot=plt)
plt.title('Normal Q-Q plot')
plt.show()

#%%
# equal variance

housing.plot(kind='scatter', x='rm', y='error', figsize=(15,8), color='green')
plt.title('Residuals vs predictor')
plt.axhline(y=0, color='red')
plt.show()

#%%




