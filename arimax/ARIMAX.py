import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dataset=pd.read_csv("energydata_complete.csv")
dataset=dataset.head(19300)
dataset['energy'] = dataset['Appliances'] + dataset['lights']
df=pd.concat([dataset['date'], dataset['energy'], dataset['T1'],dataset['RH_1'], dataset['T2'],dataset['RH_2']], axis=1)
df2=dataset['energy']
# df=df.head(19300)
# df2=df2.head(19300)

print(df2)


indexeDataset=df2
indexeDataset_logScale = np.log(indexeDataset)

import statsmodels.api as sm

df['energy']=indexeDataset_logScale
model3=sm.tsa.ARIMA(endog=df['energy'],exog=df[['T1','RH_1','T2','RH_2']],order=[5,0,1])
results3=model3.fit()


predictions_ARIMA = np.exp(results3.fittedvalues)
prediction=np.array(predictions_ARIMA.values)
print(prediction)
plt.plot(prediction)
plt.show()
plt.plot(predictions_ARIMA[19100:19300])
plt.show()


D=np.array(dataset["energy"].values)
# D=D.reshape(19300,1)
print(D.shape)

plt.plot(D[19100:19300])
plt.plot(prediction[19100:19300])
plt.show()
import math

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# a=np.sum((D - prediction)**2)
RMSE = math.sqrt(mean_squared_error(D, prediction))
MAE = mean_absolute_error(D, prediction)

# a/=17000
print(RMSE)
print(MAE)



