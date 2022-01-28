import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,6
dataset1=pd.read_csv("energydata_complete.csv")
dataset1=dataset1.head(1900)
dataset= dataset1['Appliances']
dataset=np.array(dataset)
window=1
a=[]
for i in range(len(dataset)):
    if int(i % window)==0:
        a.append(i)
a=np.array(a)
P=[]
for p in range(2):
    P.append(p)
P=np.array(P)
for i in range(len(a)):
    u=np.random.choice(P)
    t=np.random.choice(a)
    if u==0:
        dataset[t:t+window]=0
    a=np.delete(a, np.where(a==t), axis=0)
# #################
# ###imupatuion
# #################
for n in range(dataset.shape[0]):
    a=dataset
    a=np.array(a)
    p=np.where(a!=0)
    p=np.array(p)
    p=p.reshape(p.shape[1])
    if p[0]!=0:
        for i in range(p[0]):
            a[i]=a[p[0]]
    if p[len(p)-1]!=0:
        for i in range(p[len(p)-1],len(a)):
            a[i]=a[p[len(p)-1]]
    for i in range(len(p)-1):
        for j in range(p[i]+1,p[i+1]):
            a[j]=(a[p[i]] + a[p[i+1]]) / 2
    dataset=a
dataset=dataset.tolist()
print(dataset)
dataset = pd.DataFrame(dataset, columns = list("A"))
dataset.rename(columns={'A':'Appliances',}, inplace=True)
dataset=pd.concat([dataset1['date'], dataset['Appliances']], axis=1)
dataset=dataset.head(1900)



#Parse string to datatime type
dataset['date']=pd.to_datetime(dataset['date'],infer_datetime_format=True)
indexeDataset=dataset.set_index (['date'])
# print(dataset)
#plot gragh
# plt.xlabel("date")
# plt.ylabel("Energy consumtion")
# plt.plot(indexeDataset)
# plt.show()


# Determinig rolling statistics
# choon mikhastan 120 daghighe baad ro pishbini kone window=10(10*10)
rolmean=indexeDataset.rolling(window=5).mean()
rolstd=indexeDataset.rolling(window=5).std()
print(rolmean)
print(rolstd)
print(len(rolmean))

#Plot rolling statistics:
orig = plt.plot(indexeDataset, color='blue', label='Original')
mean = plt.plot(rolmean,color='red',label='Rolling Mean')
std  = plt.plot(rolstd,color='black',label='Rolling Std')
# plt.legend(loc='best')
# plt.title('Rolling Mean & Standard Deviation')
# plt.show(block=False)
# plt.show()
#
# #Perform Dickey-Fuller test:
# from statsmodels.tsa.stattools import adfuller
# print('Results of Dicky-Fuller Test:')
# dftest = adfuller(indexeDataset['Appliances'],autolag='AIC')
# dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#lag used','Number of observation used'])
# for key,value in dftest[4].items():
#     dfoutput['Critical Value (%s) '%key]=value
# print(dfoutput)

# Estimating trend
indexeDataset_logScale = np.log(indexeDataset)
# plt.plot(indexeDataset_logScale)
# plt.show()



# movingAverage = indexeDataset_logScale.rolling(window=144*7).mean()
# movingSTD = indexeDataset_logScale.rolling(window=144*7).std()

movingAverage = indexeDataset_logScale.rolling(window=5).mean()
movingSTD = indexeDataset_logScale.rolling(window=5).std()

# plt.plot(indexeDataset_logScale)
# plt.plot(movingAverage,color='red')
# plt.plot(movingSTD,color='black')
# plt.show()

datasetLogScaleMinusMovingAverage = indexeDataset_logScale - movingAverage
# datasetLogScaleMinusMovingAverage.head(12)

#Remove Nan Values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
# datasetLogScaleMinusMovingAverage.head(10)

from statsmodels.tsa.stattools import adfuller
def test_stationarity (timeseries):

    #Determining rolling statistics
    movingAverage = timeseries.rolling(window=5).mean()
    movingSTD = timeseries.rolling(window=5).std()

    # # Plot rolling statistics:
    # orig = plt.plot(timeseries, color='blue', label='Original')
    # mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    # std = plt.plot(movingSTD, color='black', label='Rolling Std')
    # plt.legend(loc='best')
    # plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)

    # # Perform Dickey-Fuller test:
    # from statsmodels.tsa.stattools import adfuller
    # print('Results of Dicky-Fuller Test:')
    # dftest = adfuller(timeseries['Appliances'], autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#lag used', 'Number of observation used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Value (%s) ' % key] = value
    # print(dfoutput)
test_stationarity(datasetLogScaleMinusMovingAverage)
#
exponentialDecayWeigthedAverage = indexeDataset_logScale.ewm(halflife=12 ,min_periods=0 , adjust= True).mean()

# plt.plot(indexeDataset_logScale)
# plt.plot(exponentialDecayWeigthedAverage, color='red')
# plt.show()

datasetLogScaleMinusMovingAverageExponentialDecayAverage = indexeDataset_logScale - exponentialDecayWeigthedAverage
test_stationarity(datasetLogScaleMinusMovingAverageExponentialDecayAverage)

datasetLogDiffShifting = indexeDataset_logScale - indexeDataset_logScale.shift()

# plt.plot(datasetLogDiffShifting)
# plt.show()
#dropped NAN value
datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)

#
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexeDataset_logScale,freq=10)
#
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
#
# plt.subplot(411)
# plt.plot(indexeDataset_logScale, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal, label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()
#
decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf =  pacf(datasetLogDiffShifting, nlags=20, method= 'ols')

#plot AcF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
plt.show()
#
#Plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()
#
from statsmodels.tsa.arima_model import ARIMA
#AR MODEL
# model = ARIMA(indexeDataset, order=(10,0,1))
model = ARIMA(indexeDataset_logScale, order=(5,0,1))
results_ARIMA = model.fit(disp=-1)
plt.plot(indexeDataset_logScale)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS:%.4f'%(sum((results_ARIMA.fittedvalues - indexeDataset_logScale["Appliances"])**2)))
plt.show()
print(results_ARIMA.fittedvalues)
print(indexeDataset_logScale)
a=sum((results_ARIMA.fittedvalues - indexeDataset_logScale["Appliances"])**2)
a/=17000
print(a)
perdictions_ARIMA = np.exp(results_ARIMA.fittedvalues)

plt.plot(indexeDataset_logScale)
plt.plot(perdictions_ARIMA)
plt.show()

# indexeDataset=np.array(indexeDataset)
# perdictions_ARIMA=np.array(perdictions_ARIMA)

# print(type(indexeDataset))
# print(type(perdictions_ARIMA))
# a=sum((dataset["Appliances"] - perdictions_ARIMA)**2)
# a/=17000
# print(a)
# print(perdictions_ARIMA["Appliances"])
# perdictions_ARIMA=np.array(perdictions_ARIMA)
# perdictions_ARIMA=perdictions_ARIMA.reshape(17000,2)
prediction=np.array(perdictions_ARIMA.values)
prediction=prediction.reshape(1900,1)
print(prediction)
# prediction=np.transpose(prediction)
# print(prediction.shape)
# print(dataset["Appliances"].values)
D=np.array(dataset["Appliances"].values)
D=D.reshape(1900,1)
print(D)

import math

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# a=np.sum((D - prediction)**2)
RMSE = math.sqrt(mean_squared_error(D, prediction))
MAE = mean_absolute_error(D, prediction)

# a/=17000
print(RMSE)
print(MAE)