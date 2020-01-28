import numpy as np 
import pandas as pd 
import datetime
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=30)
    rolstd = pd.rolling_std(timeseries, window=30)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation - 1 Day')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

#data = pd.read_csv('ohlc1dia.csv')
data = pd.read_csv('ohlc10dias.csv')
data['Datetime'].index
data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)

print(data)


x=data['Datetime']
#y1=data['Open']
y1=data['Close']
#test_stationarity(y1)  #teste estatistico de series temporais

ts_log=np.log(y1)
#moving_avg=pd.rolling_mean(ts_log,30) #ultimo 30 dias
#plt.plot(ts_log,color='blue', label='Close')
#plt.plot(moving_avg,color='red', label='Moving Average')
#plt.legend(loc='best')
#plt.title('Close versus Moving Average - 30 Days')
#note que a media movel nao esta definida para os 30 dias iniciais
#ts_log_moving_avg_diff=ts_log-moving_avg
#ts_log_moving_avg_diff.head(30)

#usando esses valores para testar
#ts_log_moving_avg_diff.dropna(inplace=True)
#test_stationarity(ts_log_moving_avg_diff)

#####################testando media movel ponderada
#expwighted_avg=pd.ewma(ts_log,halflife=30)
#plt.plot(ts_log,color='blue', label='Close')
#plt.plot(expwighted_avg,color='red', label='Weighted Average')
#plt.legend(loc='best')
#plt.title('Close versus Weighted Average - 30 Days')
#plt.show(block=False)
#ts_log_ewma_diff=ts_log-expwighted_avg

#test_stationarity(ts_log_ewma_diff)

#######################Diferenciacao

ts_log_diff=ts_log-ts_log.shift()
#plt.plot(ts_log_diff)
#ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)


###############teste decomposicao
#decomposition = seasonal_decompose(ts_log) 
#trend = decomposition.trend 
#seasonal = decomposition.seasonal 
#residual = decomposition.resid 
#plt.subplot(411) 
#plt.plot(ts_log, label='Original') 
#plt.legend(loc='best') 
#plt.subplot(412) 
#plt.plot(trend, label='Trend') 
#plt.legend(loc='best') 
#plt.subplot(413) 
#plt.plot(seasonal,label='Seasonality') 
#plt.legend(loc='best') 
#plt.subplot(414) 
#plt.plot(residual, label='Residuals')
#plt.legend(loc='best') 
#plt.tight_layout()


#################Forecast de uma TS
#lag_acf = acf(ts_log_diff, nlags=20)
#lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
##Plot ACF: 
#plt.subplot(121) 
#plt.plot(lag_acf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.title('Autocorrelation Function')
##Plot PACF: 
#plt.subplot(122)
#plt.plot(lag_pacf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
#plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray') 
#plt.title('Partial Autocorrelation Function') 
#plt.tight_layout()

#############teste ARIMA
from statsmodels.tsa.arima_model import ARIMA
#treinamento
ts=data['Close']
#plt.title('RMSE: %.4f'% sum((results_AR-ts_log)**2)/len(ts))

model = ARIMA(ts_log, order=(1,1,0)) 
results_ARIMA = model.fit(disp=-1) 
plt.plot(ts_log_diff) 
plt.plot(results_ARIMA.fittedvalues, color='red') 
ts_log_diff.fillna(value=0, inplace=True)

plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

##testes
#predictions_ARIMA_diff = pd.Series(results_AR.fittedvalues, copy=True) 
#print("teste Arima com diferencacao")
#print (predictions_ARIMA_diff.head())
#
#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum() 
#print (predictions_ARIMA_diff_cumsum.head())
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
#predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#print (predictions_ARIMA_log.head())
#predictions_ARIMA = np.exp(predictions_ARIMA_log) 
#plt.plot(ts)
#plt.plot(predictions_ARIMA) 
#plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))