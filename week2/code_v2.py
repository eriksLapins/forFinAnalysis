# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:50:41 2020

@author: User
"""

#%%
import os
os.getcwd()
os.chdir("C:/Users/User/Investment/W_Report/Python/Projects/forFinAnalysis/week2")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#%%
# Creating a dice
die = pd.DataFrame([1,2,3,4,5,6])

sum_of_dice = die.sample(2, replace=True).sum().loc[0] # Rolling a dice
print('Sum of dice is ', sum_of_dice)

# Rolling two die for 50 times
trial = 50
results = [die.sample(2,replace=True).sum().loc[0] for i in range(trial)]

results[:10] # show the 10 first trials

#%%
# Frequency
freq = pd.DataFrame(results)[0].value_counts() # Count values
sort_freq = freq.sort_index() # sort index - index is the outcome, and right column - frequency
sort_freq

#frequency plot

sort_freq.plot(kind='bar', color='blue')

# relative frequency plot

relative_freq=sort_freq/trial
relative_freq.plot(kind='bar', color='blue')

#%%
# 100 trials

trial_100 = 100
result = [die.sample(2,replace=True).sum().loc[0] for i in range(trial_100)]

# Frequency
freq = pd.DataFrame(result)[0].value_counts() # Count values
sort_freq = freq.sort_index() 

# relative frequency plot

relative_freq=sort_freq/sort_freq.sum()
relative_freq.plot(kind='bar', color='blue')

#%% distribution calculation

X_distri = pd.DataFrame(index=[2,3,4,5,6,7,8,9,10,11,12])
X_distri['Prob'] = [2,3,4,5,6,7,8,9,10,11,12]
X_distri['Prob'] = X_distri['Prob']/36 # Can divide the variable by its previous self

#%%

# Mean and Variance - for a normal distribution of discrete variables

Mean=(X_distri.index*X_distri['Prob']).sum() 
Var=(((X_distri.index-Mean)**2)*X_distri['Prob']).sum() # "**2" means to the power of two
print(Mean, Var)

#%% 

# More empirical mean and variance

trial = 20000
results = [die.sample(2, replace=True).sum().loc[0] for i in range(trial)]
# printing mean and variance for the 20000 trials

results = pd.Series(results)
print(results.mean(), results.var())
#%%

# using real data

aapl = pd.read_csv('apple.csv', index_col='Date')

# plot the returns for Aug 20212 to Aug 2013
aapl.loc['2012-08-01':'2013-08-01','Close'].plot()

# Plotting log returns on a histogram

aapl['LogReturn'] = np.log(aapl['Close']).shift(-1) - np.log(aapl['Close'])
aapl['LogReturn'].hist(bins=50)

# Looks similar to a normal distribution
#%%
from scipy.stats import norm # Scipy is a scientific python package
# We will want Probability Density Function or Cumulative Distribution Function

density = pd.DataFrame()
density['x'] = np.arange(-4,4,0.001)
density['pdf'] = norm.pdf(density['x'],0,1) # get PDF # 0 and 1 indicate 
                # STANDARD normal random variable, but you can change them to
                # get different normal random variables (here mean=0, and variance=1)
density['cdf'] = norm.cdf(density['x'],0,1) # get CDF

plt.plot(density['x'],density['pdf'])

plt.plot(density['x'],density['cdf'])

#%%

# Approximate mean and variance of the log daily return

mu= aapl['LogReturn'].mean()
sigma = aapl['LogReturn'].std(ddof=1)

print(mu,sigma)
#%%
# what is the chance of losing over 5% in a day?

denApp = pd.DataFrame()
denApp['x'] = np.arange(-0.1,0.1,0.001)
denApp['pdf'] = norm.pdf(denApp['x'],mu,sigma)
#%%
plt.ylim(0,30)
plt.plot(denApp['x'],denApp['pdf'])
plt.fill_between(x=np.arange(-0.1,-0.01,0.0001),
                 y1=norm.pdf(np.arange(-0.1, -0.01, 0.0001),mu,sigma),
                 y2=0,
                 facecolor='pink',
                 alpha=0.5)

#%%
# Write down the cumulative probability of losing more than 5 %
prob_return1 = norm.cdf(-0.05,mu,sigma)
print('The probability is ', prob_return1)

#%%

mu220 = 220*mu
sigma220 = 220**0.5*sigma
print(mu220, sigma220)

print('The probability of dropping over 40% in 220 days is ',
      norm.cdf(-0.4,mu220,sigma220))

#%%

# getting quantiales
# getting VaR - Value at Risk
norm.ppf(0.05,mu,sigma) # ppf is percent point function
# with a 5% chance, the deaily return will be worse than -2.5%


