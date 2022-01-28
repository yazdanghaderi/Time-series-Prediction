# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 18:25:58 2018

@author: diako
""""LSTM"

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 21:21:40 2018

@author: diako
"""
# importin libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import tensorflow as tf

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

dataset_tr = dataset[0:19300, :]
window = 1
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
# print(dataset)
# scaling data
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
traning_set_scaled = sc.fit_transform(dataset)

print(traning_set_scaled.shape)

# creating a data structure with 60 steps and 1 output
x_train = []
y_train = []
for i in range(30, 19100):
    #    for j in range(1,5):
    x_train.append(traning_set_scaled[i - 30:i, 0:5])
    # y_train.append(traning_set_scaled[i-59:i+1, 0])
    y_train.append(traning_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape)
print(y_train.shape)
#
##reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))
y_train = np.reshape(y_train, (y_train.shape[0],1))
print(x_train.shape)
print(y_train.shape)


# Real_enenrgy_consumtion = dataset[19100:19300, 0:5]
x_test = []
for i in range(19100, 19300):
    x_test.append(traning_set_scaled[i - 30:i, 0:5])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5))
print(x_test.shape)


tf.reset_default_graph()
num_periods = 30
inputs = 5
n_hidden = 20
output = 1
batch_size=8

X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None,output])
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=hidden,activation=tf.nn.relu) for _ in range(2)])
# rnn_output, states = tf.nn.dynamic_rnn(stacked_lstm, X, dtype = tf.float32)

cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
cell3 = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
# cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.2)
                                               , tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=0.2)
                                               , tf.contrib.rnn.DropoutWrapper(cell3, output_keep_prob=0.2)]
                                           , state_is_tuple=True)

rnn_output, states = tf.nn.dynamic_rnn(stacked_lstm, X, dtype = tf.float32)

learning_rate = 0.001
stacked_rnn_output = tf.reshape(rnn_output[:,-1], [-1, n_hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
# outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])
outputs = tf.reshape(stacked_outputs, [-1, output])
loss = tf.reduce_sum(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss, global_step=global_step)

# init = tf.global_variables_initializer()
epochs = 100

Ntr=x_train.shape[0]

starts = np.arange(0, Ntr, batch_size)
ends = np.arange(batch_size, Ntr + 1, batch_size)
if ends[-1] < Ntr:
    ends = np.append(ends, Ntr)
num_batches = len(ends)
total_batches = 0

seed = 8675309
rs = np.random.RandomState(seed)


with tf.Session() as sess:
    # init.run()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # defaults to saving all variables
    # saver.restore(sess,"checkpoint-lstm-_50miss-5feture((1,2),23,24,25,26)/LSTM-38144")
    for ep in range(epochs):
        # train
        # epoch_start = time()
        print("\tepoch:", ep)
        # print("Starting epoch " + "{:d}".format(i))
        batch = 0
        perm = rs.permutation(Ntr)
        for s, e in zip(starts, ends):
            inds = perm[s:e]

            # sess.run(training_op, feed_dict={X: x_train[s:e,:num_periods, :inputs],y: y_train[s:e,:output]})
            sess.run(training_op, feed_dict={X: x_train[inds,:num_periods, :inputs],y: y_train[inds,:output]})

            # if ep % 10 == 0:
                # mse = loss.eval(feed_dict={X: x_train[s:e,:num_periods, :inputs],y: y_train[s:e,:output]})
                # print(ep, "\tMSE", mse)

            batch += 1;
            total_batches += 1
        # print("Finishing epoch " + "{:d}".format(i) + ", took " + \
        #       "{:.3f}".format(time() - epoch_start))
        if ep % 25==0:
            y_pred = sess.run(outputs, feed_dict={X: x_test})
            print(y_pred)
            y_pred=np.array(y_pred)
            print("\ty_pred.shape:",y_pred.shape)
            # print(x_train[1901:19])
            # saved_path = saver.save(sess, 'checkpoint-lstm-_50miss-5feture((1,2),23,24,25,26)/LSTM', global_step=global_step)
            # predicted_energy_consumtion = y_pred[:30,60-1,:1]
            predicted_energy_consumtion = y_pred
            # print(predicted_energy_consumtion)
            print("\tpredicted_energy_consumtion:",predicted_energy_consumtion.shape)
            m = np.zeros((200, 4))
            V = np.c_[predicted_energy_consumtion, m]
            print(V.shape)
            predicted_energy_consumtion = sc.inverse_transform(V)
            print(predicted_energy_consumtion)
            print("\tpredicted",predicted_energy_consumtion[:, 0])
            print("\tReal",Real_energy_consumtion[:, 0])
            from sklearn.metrics import mean_absolute_error

            import math
            from sklearn.metrics import mean_squared_error
            from sklearn.metrics import mean_absolute_error

            RMSE = math.sqrt(mean_squared_error(Real_energy_consumtion[:, 0], predicted_energy_consumtion[:, 0]))
            MAE = mean_absolute_error(Real_energy_consumtion[:, 0], predicted_energy_consumtion[:, 0])
            print('Test Score: %.2f RMSE' % (RMSE))
            print('Test Score: %.2f RMSE' % (MAE))
            # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
            # print('Train Score: %.2f RMSE' % (trainScore))

            # visualizing the Results
            plt.plot(Real_energy_consumtion[:, 0], color='red', label='real data')
            plt.plot(predicted_energy_consumtion[:, 0], color='blue', label='predicted data')
            plt.title('energy_prediction')
            plt.xlabel('Time')
            plt.ylabel('value')
            plt.legend()
            plt.show()

