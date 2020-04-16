# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:17:59 2020

@author: User
"""

#%%
import pandas as pd
import numpy as np
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

