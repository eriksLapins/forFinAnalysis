# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:17:59 2020

@author: User
"""

#%%
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#%%
data = pd.DataFrame()
data['Population'] = [47,48,85,20,19,13,72,16,50,60]

#sample without replacement
sample_no_replacement = data['Population'].sample(5,replace=False)
# sample with replacement
sample_replacement = data['Population'].sample(5,replace=True)


print(sample_no_replacement)
print(sample_replacement)

#%%
# Sample Descriptors

print('Population mean is', data['Population'].mean())
print('Population variance is', data['Population'].var(ddof=0)) # ddof=0 means that we use the population number as a 
print('Population standard deviation is', data['Population'].std(ddof=0))
print('Population size is', data['Population'].shape[0])

#%%
a_sample = data['Population'].sample(10,replace=True)

print('Sample mean is', a_sample.mean())
print('Sample variance is', a_sample.var(ddof=1)) # ddof=1 means that we reduce the number by 1 degrees of freedom 
print('Sample standard deviation is', a_sample.std(ddof=1))
print('Sample size is', a_sample.shape[0])

#%%
# repeatedly try a sample of 500

sample_length = 500
# no degrees of freedom
sample_variance_collection0=[data['Population'].sample(50,
                                                       replace=True).var(ddof=0)
                             for i in range(sample_length)]
# one degrees of freedom
sample_variance_collection1=[data['Population'].sample(50,
                                                       replace=True).var(ddof=1)
                             for i in range(sample_length)]

print('Population variance is ',data['Population'].var(ddof=0))
print('Average of sample variance with n is ', 
      pd.DataFrame(sample_variance_collection0)[0].mean())
print('Average of sample variance with n-1 is ', 
      pd.DataFrame(sample_variance_collection1)[0].mean())

#%%

Fstsample = pd.DataFrame(np.random.normal(10, 5, size=30)) # create a random variable that follows normal distribution
print('Sample mean is ', Fstsample[0].mean()) # create the samples mean
print('Sample SD is ', Fstsample[0].std(ddof=1)) # create the samples standard deviation
#%%
meanlist = []
varlist = []
for t in range(1000): # 1000 samples
    sample = pd.DataFrame(np.random.normal(10, 5, size=30))
    meanlist.append(sample[0].mean()) # save 1000 sample means in a list called meanlist
    varlist.append(sample[0].var(ddof=1)) # save 1000 sample variances in a list called varlist
    
collection = pd.DataFrame()
collection['meanlist'] = meanlist
collection['varlist'] = varlist

collection['meanlist'].hist(bins=500, normed=1)
collection['varlist'].hist(bins=500, normed=1)
#%%
samplemeanlist = []
apop = pd.DataFrame([1,0,1,0,1])
for t in range(10000):
    sample = apop[0].sample(10, replace=True) #Small sample size
    samplemeanlist.append(sample.mean())
    
acollec = pd.DataFrame()
acollec['meanlist'] = samplemeanlist 

acollec['meanlist'].hist(bins=500,color='red', normed=1)

#%%

aapl = pd.read_csv('apple.csv')
aapl = aapl.loc[:'2018-01-01']
aapl['logReturn'] = np.log(aapl['Close'].shift(-1)) - np.log(aapl['Close'])
aapl['logReturn'].hist(bins=200)
#%%
# valuse for calculating the 80% confidence interval

z_left = norm.ppf(0.1)
z_right = norm.ppf(0.9)
sample_mean = aapl['logReturn'].mean()
sample_std = aapl['logReturn'].std(ddof=1)/(aapl.shape[0])**0.5

interval_left = sample_mean+z_left*sample_std
interval_right = sample_mean+z_right*sample_std
print('Sample mean is ', sample_mean)
print('**************************************')
print('80% confidence interval is ')
print(interval_left, interval_right)
#%%
plt.title("Close Price of Apple from 2007 to 2019", size=30)
plt.xlabel("Time", size=20)
plt.ylabel("US $", size=20)
plt.plot(aapl.loc[:,'Close'])

#%% 
# Hypothesis testing

# getting z hat
xbar = aapl['logReturn'].mean()
s = aapl['logReturn'].std(ddof=1)
n = aapl['logReturn'].shape[0]
zhat = (xbar-0)/(s/(n**0.5))
print(zhat)
#%%
# two tail test
alpha=0.05
zleft = norm.ppf(alpha/2,0,1)
zright = -zleft
print(zleft, zright)
print('At the significance level of ', alpha)
print('Shall we reject?:', zhat>zright or zhat<zleft)
#%%
# one tail test
alpha=0.05
zright = norm.ppf(1-alpha,0,1)
zright = -zleft
print(zleft, zright)
print('At the significance level of ', alpha)
print('Shall we reject?:', zhat>zright)

#%%
# Calculate p-value for two tails test

alpha = 0.05
p = 1-(norm.cdf(abs(zhat),0,1))
print('At the significance level of ', alpha)
print('Shall we reject: ', p< alpha)

#%%
aapl['logReturn'].plot(figsize=(20, 8))
plt.axhline(0, color='red')
plt.show()



