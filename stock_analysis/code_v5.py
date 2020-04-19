# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:20:08 2020

@author: User
"""

import os
os.getcwd()
os.chdir("C:/Users/User/Investment/W_Report/Python/Projects/forFinAnalysis/stock_analysis")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf # used for the regression

#%%

# data
aord=pd.read_csv('aord.csv')
nikkei=pd.read_csv('nikkei.csv')
hsi=pd.read_csv('hsi.csv')
daxi=pd.read_csv('daxi.csv')
cac40=pd.read_csv('cac40.csv')
sp500=pd.read_csv('sp500.csv')
dji=pd.read_csv('dji.csv')
nasdaq=pd.read_csv('nasdaq.csv')
spy=pd.read_csv('spy.csv')

#%%

spy.head() # will use only open to simpolify

# response variable : SPY's tomorrows open price minus todays open

# 8 predictors: 1 day lag variables for us market (open-open last day) for us markets
#               open-open last day for EU markets (If you have intraday data, it would
                        # be better to have midday price at the point when US market opens)
#               close-open for asian markets

# generate and empty dataframe that indexes by spy
indicepanel=pd.DataFrame(index=spy.index)
indicepanel['spy']=spy['Open'].shift(-1)-spy['Open']
#lagged spy
indicepanel['spy_lag1']=indicepanel['spy'].shift(1) # to create a lagged variable
# us markets
indicepanel['sp500']=sp500['Open']-sp500['Open'].shift(1)
indicepanel['nasdaq']=nasdaq['Open']-nasdaq['Open'].shift(1)
indicepanel['dji']=dji['Open']-dji['Open'].shift(1)
# eu markets
indicepanel['cac40']=cac40['Open']-cac40['Open'].shift(1)
indicepanel['daxi']=daxi['Open']-daxi['Open'].shift(1)
# asian markets
indicepanel['aord']=aord['Close']-aord['Open']
indicepanel['hsi']=hsi['Close']-hsi['Open']
indicepanel['nikkei']=nikkei['Close']-nikkei['Open']

indicepanel['Price']=spy['Open']

indicepanel.head() # empty first row because of lag

#%%
################
#Data munging
################

#%%
indicepanel=indicepanel.fillna(method='ffill') # use forward fill to get rid of NaN values
indicepanel.isnull().sum() # show how many NaN's we have
indicepanel=indicepanel.dropna() # drop the NaN's
indicepanel.isnull().sum() # show how many NaN's we have

# GETTING OUT TO CSV
indicepanel.to_csv('indicepanel.csv') # save the dataframe to a csv file

indicepanel.shape

#%%
###################
# Data splitting
####################

#%%
Train=indicepanel.iloc[-2000:-1000,:] # Training set (1000 days before the test data days)
Test=indicepanel.iloc[-1000:,:] # Test set to see if trained set can reasonably predict
                                # We test the trained model on the last 1000 days
print(Train.shape, Test.shape)

#%%

from pandas.plotting import scatter_matrix
sm=scatter_matrix(Train, figsize=(10,10)) # create a scatter plot to check initial relationships

# Correlations between spy and other variables

Train.iloc[:,:-1].corr()['spy'] # no really high correlations indicate the high noisiness of markets
# but it seems that european and asian markets have higher correlation with spy

#%%

# Regression model

formula = 'spy~spy_lag1+sp500+nasdaq+dji+cac40+daxi+aord+nikkei+hsi'
lm=smf.ols(formula, data=Train).fit()
lm.summary()
# most variables are useless in predicting

# Multicollinearity

Train.iloc[:,:-1].corr() # we see quite high correlations between the variables

#%%
####################
# Making prediction
###################
#%%
Train['PredictedY']=lm.predict(Train)
Test['PredictedY']=lm.predict(Test)

plt.scatter(Train['spy'], Train['PredictedY'])

#%%
###################
# Model evaluation:
##################
#%%
# Some statistics for tests:
            # RMSE - root mean square error
            # Adjusted R^2

def adjustedMetric(data, model, model_k, yname): # we can save the computations in a function
                # data - data
                # model - model name
                # model_k - number of predictors
                # yname - column name of our response variable
    data['yhat']=model.predict(data)
    SST=((data[yname]-data[yname].mean())**2).sum()
    SSR=((data['yhat']-data[yname].mean())**2).sum()
    SSE=((data[yname]-data['yhat'])**2).sum()
    r2=SSR/SST
    adjustR2=1-(1-r2)*(data.shape[0]-1)/(data.shape[0]-model_k-1)
    RMSE=(SSE/(data.shape[0]-model_k-1))**0.5
    return adjustR2, RMSE

#%%
    
print('Adjusted R2 and RMSE on Train: ', adjustedMetric(Train,lm,9,'spy'))
print('Adjusted R2 and RMSE on Test: ', adjustedMetric(Test,lm,9,'spy'))

#%%
# Create another functino that compares R2 and RMSE betwen Train and Test sets

def assessTable(test, train, model, model_k, yname):
    r2test,RMSEtest=adjustedMetric(test,model,model_k,yname)
    r2train,RMSEtrain=adjustedMetric(train,model,model_k,yname)
    assessment=pd.DataFrame(index=['R2','RMSE'],columns=['Train', 'Test'])
    assessment['Train']=[r2train,RMSEtrain]
    assessment['Test']=[r2test,RMSEtest]
    return assessment

#%%
    # assessment of the R2 and RMSE between test and train sets

assessTable(Test,Train,lm,9,'spy') 

# If RMSE and adjusted R2 in training and test sets differ dramatically:
    # we cannot apply this model to stock markets - the model is crap
# Our R2 is quite low but improving, and our RMSE is increased, which is a bit worse:
        # however, our model does not seem to be overfitting and we could apply it
# Our R2 is just 1.5% but in stock markets it is not that bad

#%%
# Profit of signal-based strategy (Train data)

Train['Order']=[1 if sig>0 else -1 for sig in Train['PredictedY']] # buy 1 share
    # if the change in the price from today's open to tomorrow's is positive,
    # otherwise, short 1 share
Train['Profit']=Train['spy']*Train['Order']

Train['Wealth']=Train['Profit'].cumsum()
print('Total profit made in train: ', Train['Profit'].sum())

#%%
# Plotting the difference between buy and hold vs signal-based strategy
plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Train')
plt.plot(Train['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(Train['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()

#%%

# Similarly lets do it for test
Test['Order']=[1 if sig>0 else -1 for sig in Test['PredictedY']]
Test['Profit']=Test['spy']*Test['Order']

Test['Wealth']=Test['Profit'].cumsum()
print('Total profit made in Test: ', Test['Profit'].sum())

#%%
# Same plot but for test model
plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Train')
plt.plot(Test['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(Test['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()

#%%

# Sharpe ratio calculation
# Sharpe ratio in Train data
Train['Wealth']=Train['Wealth']+Train.loc[Train.index[0],'Price'] #Add the initial investment
                        # by the second part of this equation
Test['Wealth']=Test['Wealth']+Test.loc[Test.index[0],'Price']

Train['Return']=np.log(Train['Wealth'])-np.log(Train['Wealth'].shift(1))
dailyr=Train['Return'].dropna() # similarly can do it for Test data

print('Daily SR is ', dailyr.mean()/dailyr.std(ddof=1))
print('Yearly SR is ', (252**0.5)*dailyr.mean()/dailyr.std(ddof=1))

# Sharpe Ratio in Test data
Test['Return'] = np.log(Test['Wealth']) - np.log(Test['Wealth'].shift(1))
dailyr = Test['Return'].dropna()

print('Daily Sharpe Ratio is ', dailyr.mean()/dailyr.std(ddof=1))
print('Yearly Sharpe Ratio is ', (252**0.5)*dailyr.mean()/dailyr.std(ddof=1))

#%%

# Maximum drawdown, i.e., the maximum percentage decline in the strategy from the
    # historical peak profit at each point in time

# drawdown = (maximum-wealth)/maximum

Train['Peak']=Train['Wealth'].cummax()
Train['Drawdown']=(Train['Peak']-Train['Wealth'])/Train['Peak']

Test['Peak']=Test['Wealth'].cummax()
Test['Drawdown']=(Test['Peak']-Test['Wealth'])/Test['Peak']

print('Maximum train drawdown is ', Train['Drawdown'].max())
print('Maximum test drawdown is ', Test['Drawdown'].max()) # Test data has way more drawdown

# Not a very consistent model with this data
# Have to consider that there is a bid-ask spread which may eat up all the profit






