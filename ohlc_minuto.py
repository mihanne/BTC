import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from datetime import datetime
from pandas import DataFrame, Series 

df = DataFrame() 

df = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv")
print("dados")

df.timestamp = pd.to_datetime(df.timestamp)
df = df.set_index(["timestamp"])
df_ohlc = df.resample("180Min")    
df_ohlc.agg({'Open': np.fist,'High':np.max})
print (df_ohlc)
