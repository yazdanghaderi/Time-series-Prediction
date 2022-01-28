import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import tensorflow as tf


dataframe = read_csv('energydata_complete.csv', usecols=[1, 2, 3, 4, 5, 6], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset = dataset[0:19600, 0:6]

Real = np.array(dataset)
Real[:, 0] = np.add(Real[:, 0], Real[:, 1])
Real_energy_consumtion = np.delete(Real, 1, axis=1)
Real_energy_consumtion = Real_energy_consumtion[19100:19300, 0:5]

dataset = np.array(dataset)
dataset[:, 0] = np.add(dataset[:, 0], dataset[:, 1])
dataset = np.delete(dataset, 1, axis=1)

dataset_tr = dataset[0:19300, :]
window = 1
a = []

seed = 8
rs = np.random.RandomState(seed)

for i in range(dataset_tr.shape[0]):
    if int(i % window) == 0:
        a.append(i)
a = np.array(a)
#     print(type(a))

P = []
# for p in range(dataset_tr.shape[1]):
#     P.append(p)
# P = np.array(P)


for p in range(1):
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
    # dataset[t:t + window, u2] = 0
    # dataset[t:t + window, u3] = 0
    a = np.delete(a, np.where(a == t), axis=0)
#         print("\ta:", a)
#     print(dataset_tr)

# import numpy as np
# a=[0,0,0,3,1,1,5,4,4,6,0,0,0,1,2,5,6,0,4,0,0,7,8,9,0,0,0,0
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


# #################
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

dataset[0:19300, :] = dataset_tr

# Real_energy_consumtion = dataset[19100:19300,0]
#scaling data
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
traning_set_scaled=sc.fit_transform(dataset)
#print(traning_set_scaled)
# print(traning_set_scaled.shape)

#creating a data structure with 60 steps and 1 output
x_train=[]
y_train=[]
for i in range(30,19100):
#    for j in range(1,5):
    x_train.append(traning_set_scaled[i-30:i,0:5])
    # y_train.append(traning_set_scaled[i:i+5,0])
    y_train.append(traning_set_scaled[i,0])
x_train,y_train=np.array(x_train) ,np.array(y_train)
y_train=np.reshape(y_train,(y_train.shape[0],1))########################
# print(y_train.shape)
##reshaping
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],5))
# print(x_train.shape)
#########################
a=np.zeros((19070,30*5))
for s in range(0,19070):
    for i in range(0,30):
        for j in range(0,5):
            a[s,((i*5)+j)]=x_train[s,i,j]
x_train=a
# print(x_train.shape)


 # Real_stock_price = dataset[19000:19300, 0:26]
x_test = []
for i in range(19100, 19300):
    x_test.append(traning_set_scaled[i - 30:i, 0:5])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5))
# print(x_test.shape)
# ###################################
a=np.zeros((200,30*5))
for s in range(0,200):
    for i in range(0,30):
        for j in range(0,5):
            a[s,((i*5)+j)]=x_test[s,i,j]
x_test=a

# ######################## prepare the data ########################
# X, y = load_linnerud(return_X_y = True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = False, test_size = 0.33)
#
#
# ######################## set learning variables ##################
# epochs = 10
# batch_size = 3



tf.reset_default_graph()
num_periods = 30
inputs = 5
n_hidden = 20
output = 1
batch_size=8
learning_rate = 0.01


#
#
# ######################## set some variables #######################
X = tf.placeholder(tf.float32, [None, 150], name = 'X')   # 3 features
Y = tf.placeholder(tf.float32, [None, 1], name = 'Y')   # 3 outputs
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

#

# # input-to-hidden layer1
W1 = tf.Variable(tf.random_normal([150, 40], stddev=(1 / tf.sqrt(float(150)))), name='W1')
b1 = tf.Variable(tf.random_normal([40]), name='b1')

W2 = tf.Variable(tf.random_normal([40, 40], stddev=(1 / tf.sqrt(float(40)))), name='W2')
b2 = tf.Variable(tf.random_normal([40]), name='b2')

# W3 = tf.Variable(tf.random_normal([40, 40], stddev=(1 / tf.sqrt(float(40)))), name='W3')
# b3 = tf.Variable(tf.random_normal([40]), name='b3')

# hidden layer1-to-output
W3 = tf.Variable(tf.random_normal([40, 1], stddev=(1 / tf.sqrt(float(40)))), name='W3')
b3 = tf.Variable(tf.random_normal([1]), name='b3')
# ######################## Activations, outputs ######################
# output hidden layer 1
hidden_out1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
hidden_out2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out1, W2), b2))
# hidden_out3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out2, W3), b3))
# hidden_out3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out2, W3), b3))

# total output
y_ = tf.nn.sigmoid(tf.add(tf.matmul(hidden_out2, W3), b3))
# #
# #
# # ####################### Loss Function  #########################
# mse = tf.losses.mean_squared_error(Y, y_)
# #
# #
# # ####################### Optimizer      #########################
# optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
# #
# training_op = optimizer.minimize(mse)


outputs = tf.reshape(y_, [-1, 1])
# loss = tf.reduce_sum(tf.square(outputs - Y))
loss = tf.losses.mean_squared_error(Y,outputs)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss, global_step=global_step)







#
# ###################### Initialize, Accuracy and Run #################
# initialize variables
# init_op = tf.global_variables_initializer()

##accuracy for the test set
# accuracy = tf.reduce_mean(tf.square(tf.subtract(y, y_))) # or could use tf.losses.mean_squared_error
#
epochs = 101


Ntr=x_train.shape[0]

starts = np.arange(0, Ntr, batch_size)
ends = np.arange(batch_size, Ntr + 1, batch_size)
if ends[-1] < Ntr:
    ends = np.append(ends, Ntr)
num_batches = len(ends)
total_batches = 0

# seed = 8675309
# rs = np.random.RandomState(seed)


with tf.Session() as sess:
    # init.run()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()  # defaults to saving all variables
    # saver.restore(sess,"checkpoint-mlp-Random-miss-5feture((1,2),23,24,25,26)/mlp-50064")
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
            sess.run(training_op, feed_dict={X: x_train[inds,:num_periods*inputs],Y: y_train[inds,:output]})

            # if ep % 10 == 0:
                # mse = loss.eval(feed_dict={X: x_train[s:e,:num_periods, :inputs],y: y_train[s:e,:output]})
                # print(ep, "\tMSE", mse)

            batch += 1;
            total_batches += 1
        # print("Finishing epoch " + "{:d}".format(i) + ", took " + \
        #       "{:.3f}".format(time() - epoch_start))
        if ep % 25==0:
            y_pred = sess.run(y_, feed_dict={X: x_test})
            # print(y_pred)
            y_pred=np.array(y_pred)
            # print("\ty_pred.shape:",y_pred.shape)
            # saved_path = saver.save(sess, 'checkpoint-mlp-Random-miss-5feture((1,2),23,24,25,26)/mlp', global_step=global_step)
            print("\tglobal-step:",sess.run(global_step))
            predicted_energy_consumtion = y_pred
            # print(predicted_energy_consumtion)
            # print("\tpredicted_energy_consumtion:",predicted_energy_consumtion.shape)
            m = np.zeros((200, 4))
            V = np.c_[predicted_energy_consumtion, m]
            # print(V.shape)
            predicted_energy_consumtion = sc.inverse_transform(V)
            # print(predicted_energy_consumtion)
            # print("\tpredicted",predicted_energy_consumtion[:, 0])
            # print("\tReal",Real_energy_consumtion[:, 0])

            import math
            from sklearn.metrics import mean_squared_error
            from sklearn.metrics import mean_absolute_error


            RMSE = math.sqrt(mean_squared_error(Real_energy_consumtion[:, 0], predicted_energy_consumtion[:, 0]))
            MAE = mean_absolute_error(Real_energy_consumtion[:, 0], predicted_energy_consumtion[:, 0])
            print('Test Score: %.2f RMSE' % (RMSE))
            print('Test Score: %.2f MAE' % (MAE))
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
