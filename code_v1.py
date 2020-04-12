# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:36:24 2020

@author: User
"""

import os
os.getcwd()
os.chdir("C:/Users/User/Investment/W_Report/Python/Projects/forFinAnalysis")

# import packages for analysis

import pandas as pd

# import data as a data frame

fb = pd.DataFrame.from_csv('data/facebook.csv')
ms = pd.DataFrame.from_csv('data/microsoft.csv')

# show the type of data

print(type(fb)) # .DataFrame shows that it is a data frame