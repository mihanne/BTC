# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 11:52:17 2019

@author: hanne
"""
# importar bibliotecas

import numpy as np 
import pandas as pd
from pandas import DataFrame
from pandas import Series
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_csv('ohlc1dia.csv', header=0,index_col='Datetime')


data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)
data['Volume_(BTC)'].fillna(value=0, inplace=True)
print (data)

#data['CloseOpen']=(data['Close']-data['Open'])
s1 = pd.Series(data['Open'])
data['Media'] = pd.rolling_mean(s1, 30)
data['Desvio_Padrao']= pd.rolling_std(s1,30)


#for item in data:
  # data.sort_index(inplace=True)
 #  item['1_day_mean'] = data['Close'].rolling(20).mean()
 #  item['1_day_volatility'] = data['Close'].rolling(window=20).std()

data[['Open','Media','Desvio_Padrao']].plot(figsize=(10,8));
plt.title('Bitcoin - Valor de Abertura (Open) com variação de um 1 dia')
plt.ylabel('Preço $')
plt.show()

print(data['Open'].resample('A').mean())
print(data['Open'].resample('A').apply(lambda x: x[-1]))