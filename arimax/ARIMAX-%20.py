import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
# import tensorflow as tf

dataframe = read_csv('energydata_complete.csv', usecols=[1 ,2 ,3, 4, 5, 6], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

Real = np.array(dataset)
Real[:, 0] = np.add(Real[:, 0], Real[:, 1])
Real_energy_consumtion = np.delete(Real, 1, axis=1)
Real_energy_consumtion = Real_energy_consumtion[19100:19300, 0:5]
# dataset_test=dataset[19100:19300, :]

dataset = np.array(dataset)
dataset[:,0]=np.add(dataset[:,0],dataset[:,1])
dataset = np.delete(dataset, 1, axis=1)
dataset = dataset[0:19300, :]
LL=dataset[:,0:1]
LL=LL.tolist()




dataset_tr = dataset[0:19300, :]
window = 35
a = []

for i in range(dataset_tr.shape[0]):
    if int(i % window) == 0:
        a.append(i)
a = np.array(a)
#     print(type(a))

P = []
# for p in range(dataset_tr.shape[1]):
#     P.append(p)
# P = np.array(P)

for p in range(0,dataset_tr.shape[1]):
    P.append(p)
P = np.array(P)


for i in range(len(a)):
    u1 = np.random.choice(P)
    u2 = np.random.choice(P)
    u3 = np.random.choice(P)
    # print("\tu:", u)
    # print("\ta:", a)
    t = np.random.choice(a)
    #         print("\tt:", t)
    dataset[t:t + window, u1] = 0
    dataset[t:t + window, u2] = 0
    dataset[t:t + window, u3] = 0
    a = np.delete(a, np.where(a == t), axis=0)
#         print("\ta:", a)
#     print(dataset_tr)

count = 0  # number of zero in dataset
for i in range(dataset_tr.shape[1]):
    for j in range(dataset_tr.shape[0]):
        if dataset_tr[j, i] == 0:
            count += 1
tota = (dataset_tr.shape[0]) * (dataset_tr.shape[1])
miss_rate = count / tota
trainfrac=(int(miss_rate*100))/100
print("\tmiss_rate:", miss_rate)
print("\ttrainfrac:", trainfrac)
dataset[0:19300, :] = dataset_tr


 #################
# ###imupatuion
# #################

for n in range(dataset.shape[1]):
    a=dataset[:,n]
    a=np.array(a)
    # for i in range(len(a)):
    p=np.where(a!=0)
    p=np.array(p)
    p=p.reshape(p.shape[1])
    # print(p)
    if p[0]!=0:
        for i in range(p[0]):
            a[i]=a[p[0]]
    # print(a)
    if p[len(p)-1]!=0:
        for i in range(p[len(p)-1],len(a)):
            a[i]=a[p[len(p)-1]]
    # print(a)

    for i in range(len(p)-1):
        for j in range(p[i]+1,p[i+1]):
            a[j]=(a[p[i]] + a[p[i+1]]) / 2
    # print(a)
    dataset[:,n]=a


print(dataset.shape)
# df = pd.DataFrame(ex_dict, columns=columns, index=['Appliances','lights','T1','RH_1','T2','RH_2'])
# columns
columns_new = ['energy','T1','RH_1','T2','RH_2']

# pass in array and columns
df=pd.DataFrame(dataset, columns = columns_new)


# dataset['energy'] = dataset['Appliances'] + dataset['lights']
# df=pd.concat([dataset['date'], dataset['energy'], dataset['T1'],dataset['RH_1'], dataset['T2'],dataset['RH_2']], axis=1)
df2=df['energy']
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


# D=np.array(dataset["energy"].values)
# D=D.reshape(19300,1)
D=np.array(LL)
print(D)

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



