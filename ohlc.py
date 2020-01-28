import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from datetime import datetime
from pandas import DataFrame, Series 


#Import data
data = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv")
print("data")
#Transform timestamp to datetime.
data['Datetime'] = pd.to_datetime(data['Timestamp'].apply(lambda date: datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')))

#Prepare Dataframe.
data = data.set_index('Datetime')
data = data.drop(["Timestamp"], axis=1)

# Set parameters for ohlc resampling.
ohlc_interval = "180Min"
#Generate ohlc
#data = data.set_index(["Timestamp"])
#df_ohlc = data.resample("180Min").max()    
#print (df)
#df_ohlc = data.resample('180Min').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume_(BTC)': 'sum'})

df_ohlc = data.resample('180Min')

#dfy = gb.agg({
 #'cat': np.count_nonzero,
 #'col1': [np.sum, np.mean, np.std],
 #'col2': [np.min, np.max]
#}) 

df_ohlc.aggregate({'Open': np.fist,'High':np.max})
#({'result' : lambda x: x.mean() / x.std(),
 #          'total' : np.sum})


df_ohlc.to_csv('ohlc180m.csv')
print(df_ohlc)
