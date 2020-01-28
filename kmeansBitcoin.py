# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:15:38 2019

@author: hanne
"""
from sklearn.cluster import KMeans
from pandas import DataFrame
import numpy as np 
import pandas as pd 
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt



#define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

print('Mostrando os dados...')
data = pd.read_csv('bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv', parse_dates=True, date_parser=dateparse, index_col=[0])
data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Volume_(Currency)'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)

df=DataFrame(data,columns=['High','Low'])
modelo = KMeans(n_clusters=3).fit(df)
centroids=modelo.cluster_centers_
print(centroids)
distance=modelo.fit_transform(df)
print(distance)
print("Erro quadratico")
print(modelo.inertia_) #erro quadratico medio
x=list(range(0,2629,1))

plt.scatter(df['High'], df['Low'], c= modelo.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
#plt.scatter(dados[:, 0], dados[:, 1])
plt.show()
