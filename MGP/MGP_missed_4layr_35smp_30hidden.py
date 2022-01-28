import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
from util1 import pad_rawdata, SE_kernel, OU_kernel, dot, CG, Lanczos, block_CG, block_Lanczos


def sim_dataset(num_rowdata, M, trainfrac):
    # true_Kfs, true_noises, true_lengths = gen_MGP_params(M)
    # end_times = np.random.uniform(10, 50, num_rowdata)  # last observation time of the encounter
    end_times = []
    for _ in range(num_rowdata):
        end_times.append(length_of_any_row_data)
    end_times = np.array(end_times)
    # print(end_times.shape)## (num_rowdata, )

    # num_obs_times = np.random.poisson(end_times,
    #                                   num_encs) + 3  # number of observation time points per encounter, increase with  longer series
    num_obs_times = np.zeros(num_rowdata) + 30  ##shape=(num_rowdata, )

    # num_obs_values = np.array(num_obs_times * M * trainfrac, dtype="int")
    # num_obs_values = np.array(num_obs_times * M , dtype="int")  ###shape=(num_rowdata, )

    # number of inputs to RNN. will be a grid on integers, starting at 0 and ending at next integer after end_time
    # num_rnn_grid_times = np.array(np.floor(end_times) + 1, dtype="int")
    num_rnn_grid_times = end_times  ###shape=(num_rowdata, )
    rnn_grid_times = []
    # labels = rs.binomial(1, pos_class_rate, num_encs)  # pos_class_rate=probability of one(1)
    T = [];  # actual observation times
    Y = [];
    ind_kf = [];
    ind_kt = []  # actual data; indices pointing to which lab, which time

    print('Simming data...')

    # kkk = []
    # kk = []
    # for i in range(M):
    #     kk.append(i)
    # for j in range(60):
    #     kkk.append(kk)
    # kkk = np.array(kkk)
    num_obs_values=[]
    for i in range(num_rowdata):
        if i % 500 == 0:
            print('%d/%d' % (i, num_rowdata))
        # obs_times = np.insert(np.sort(np.random.uniform(0, end_times[i], num_obs_times[i] - 1)), 0, 0)
        obs_times = np.linspace(0, end_times[i] - 1, num_obs_times[i])
        T.append(obs_times)
        # l = labels[i]
        y_i, ind_kf_i, ind_kt_i = sim_multitask_GP(x_train[i, :, :], x_train.shape[2], trainfrac,i)

        num_obs_values.append(len(y_i))  ###shape=(num_rowdata, )

        Y.append(y_i);
        ind_kf.append(ind_kf_i);
        ind_kt.append(ind_kt_i)
        rnn_grid_times.append(np.arange(num_rnn_grid_times[i]))

    T = np.array(T)  ###shape=(num_rowdata,length_of_any_row_data)
    Y = np.array(Y);
    ind_kf = np.array(ind_kf);
    ind_kt = np.array(ind_kt)
    # meds_on_grid = np.array(meds_on_grid)
    rnn_grid_times = np.array(rnn_grid_times)
    num_obs_values=np.array(num_obs_values)
    return (num_obs_times, num_obs_values, num_rnn_grid_times, rnn_grid_times,
            T, Y, ind_kf, ind_kt)


def sim_multitask_GP(observation, K_f, trainfrac,i):
    """
    draw from a multitask GP.

    we continue to assume for now that the dim of the input space is 1, ie just time

    M: number of tasks (labs/vitals/time series)

    train_frac: proportion of full M x N data matrix Y to include

    """
    M = x_train.shape[2]

    N = len(observation)
    n = N * M
    # K_t = OU_kernel_np(length, times)  # just a correlation function
    # Sigma = np.diag(noise_vars)
    #
    # K = np.kron(K_f, K_t) + np.kron(Sigma, np.eye(N)) + 1e-6 * np.eye(n)
    # L_K = np.linalg.cholesky(K)

    # y = np.dot(L_K, np.random.normal(0, 1, n))  # Draw normal

    y = observation  ####x_train[i,:,:]
    y = y.flatten('F')

    # print(y.shape)
    # get indices of which time series and which time point, for each element in y
    ind_kf = np.tile(np.arange(M), (N, 1)).flatten('F')  # vec by column
    ind_kx = np.tile(np.arange(N), (M, 1)).flatten()

     # randomly dropout some fraction of fully observed time series
    # n_train = int(trainfrac * n)
    # perm = np.random.permutation(n)
    # train_inds = perm[:n_train]

    if (i < 19070 ):
        n_train = int(trainfrac * n)
    else:
        n_train = int(n*trainfrac)

    train_inds = []
    for i in range(len(y)):
        if y[i] != 0:
            train_inds.append(i)
    train_inds = np.array(train_inds)
    perm = np.random.permutation(train_inds)
    #     train_inds = perm[:n_train]

    train_inds = perm[:n_train]
    #     train_inds=train_inds[:n_train]

    y_ = y[train_inds]
    ind_kf_ = ind_kf[train_inds]
    ind_kx_ = ind_kx[train_inds]

    return y_, ind_kf_, ind_kx_


def draw_GP(Yi, Ti, Xi, ind_kfi, ind_kti):
    """
    given GP hyperparams and data values at observation times, draw from
    conditional GP

    inputs:
        length,noises,Lf,Kf: GP params
        Yi: observation values
        Ti: observation times
        Xi: grid points (new times for rnn)
        ind_kfi,ind_kti: indices into Y
    returns:
        draws from the GP at the evenly spaced grid times Xi, given hyperparams and data
    """
    ny = tf.shape(Yi)[0]
    K_tt = OU_kernel(length, Ti, Ti)
    D = tf.diag(noises)

    grid_f = tf.meshgrid(ind_kfi, ind_kfi)  # same as np.meshgrid
    Kf_big = tf.gather_nd(Kf, tf.stack((grid_f[0], grid_f[1]), -1))

    grid_t = tf.meshgrid(ind_kti, ind_kti)
    Kt_big = tf.gather_nd(K_tt, tf.stack((grid_t[0], grid_t[1]), -1))

    Kf_Ktt = tf.multiply(Kf_big, Kt_big)

    DI_big = tf.gather_nd(D, tf.stack((grid_f[0], grid_f[1]), -1))
    DI = tf.diag(tf.diag_part(DI_big))  # D kron I

    # data covariance.
    # Either need to take Cholesky of this or use CG / block CG for matrix-vector products
    Ky = Kf_Ktt + DI + 1e-6 * tf.eye(ny)

    ### build out cross-covariances and covariance at grid

    nx = tf.shape(Xi)[0]

    K_xx = OU_kernel(length, Xi, Xi)
    K_xt = OU_kernel(length, Xi, Ti)

    ind = tf.concat([tf.tile([i], [nx]) for i in range(M)], 0)
    grid = tf.meshgrid(ind, ind)
    Kf_big = tf.gather_nd(Kf, tf.stack((grid[0], grid[1]), -1))
    ind2 = tf.tile(tf.range(nx), [M])
    grid2 = tf.meshgrid(ind2, ind2)
    Kxx_big = tf.gather_nd(K_xx, tf.stack((grid2[0], grid2[1]), -1))

    K_ff = tf.multiply(Kf_big, Kxx_big)  # cov at grid points

    full_f = tf.concat([tf.tile([i], [nx]) for i in range(M)], 0)
    grid_1 = tf.meshgrid(full_f, ind_kfi, indexing='ij')
    Kf_big = tf.gather_nd(Kf, tf.stack((grid_1[0], grid_1[1]), -1))
    full_x = tf.tile(tf.range(nx), [M])
    grid_2 = tf.meshgrid(full_x, ind_kti, indexing='ij')
    Kxt_big = tf.gather_nd(K_xt, tf.stack((grid_2[0], grid_2[1]), -1))

    K_fy = tf.multiply(Kf_big, Kxt_big)

    # now get draws!
    y_ = tf.reshape(Yi, [-1, 1])
    # Mu = tf.matmul(K_fy,CG(Ky,y_)) #May be faster with CG for large problems
    sess = tf.Session()
    print(Ky)

    Ly = tf.cholesky(Ky)
    Mu = tf.matmul(K_fy, tf.cholesky_solve(Ly, y_))

    # TODO: it's worth testing to see at what point computation speedup of Lanczos algorithm is useful & needed.
    # For smaller examples, using Cholesky will probably be faster than this unoptimized Lanczos implementation.
    # Likewise for CG and BCG vs just taking the Cholesky of Ky once
    """
    #Never need to explicitly compute Sigma! Just need matrix products with Sigma in Lanczos algorithm
    def Sigma_mul(vec):
        # vec must be a 2d tensor, shape (?,?) 
        return tf.matmul(K_ff,vec) - tf.matmul(K_fy,block_CG(Ky,tf.matmul(tf.transpose(K_fy),vec))) 

    def small_draw():   
        return Mu + tf.matmul(tf.cholesky(Sigma),xi)
    def large_draw():             
        return Mu + block_Lanczos(Sigma_mul,xi,n_mc_smps) #no need to explicitly reshape Mu

    BLOCK_LANC_THRESH = 1000
    draw = tf.cond(tf.less(nx*M,BLOCK_LANC_THRESH),small_draw,large_draw)     
    """

    xi = tf.random_normal((nx * M, n_mc_smps))
    Sigma = K_ff - tf.matmul(K_fy, tf.cholesky_solve(Ly, tf.transpose(K_fy))) + 1e-6 * tf.eye(tf.shape(K_ff)[0])
    draw = Mu + tf.matmul(tf.cholesky(Sigma), xi)
    draw_reshape = tf.transpose(tf.reshape(tf.transpose(draw), [n_mc_smps, M, nx]), perm=[0, 2, 1])
    return draw_reshape


def get_GP_samples(Y, T, X, ind_kf, ind_kt, num_obs_times, num_obs_values,
                   num_rnn_grid_times):
    """
    returns samples from GP at evenly-spaced gridpoints
    """

    grid_max = tf.shape(X)[1]
    # with tf.Session() as sess:
    # print(sess.run(grid_max))
    # print("grid_max:%s"%grid_max)
    # print("grid_max:%s"%tf.shape(X)[2])
    Z = tf.zeros([0, grid_max, input_dim])
    # print()
    N = tf.shape(T)[0]  # number of observations

    # setup tf while loop (have to use this bc loop size is variable)
    def cond(i, Z):
        return i < N

    def body(i, Z):
        Yi = tf.reshape(tf.slice(Y, [i, 0], [1, num_obs_values[i]]), [-1])
        Ti = tf.reshape(tf.slice(T, [i, 0], [1, num_obs_times[i]]), [-1])
        ind_kfi = tf.reshape(tf.slice(ind_kf, [i, 0], [1, num_obs_values[i]]), [-1])
        ind_kti = tf.reshape(tf.slice(ind_kt, [i, 0], [1, num_obs_values[i]]), [-1])
        Xi = tf.reshape(tf.slice(X, [i, 0], [1, num_rnn_grid_times[i]]), [-1])
        X_len = num_rnn_grid_times[i]

        GP_draws = draw_GP(Yi, Ti, Xi, ind_kfi, ind_kti)
        pad_len = grid_max - X_len  # pad by this much
        padded_GP_draws = tf.concat([GP_draws, tf.zeros((n_mc_smps, pad_len, M))], 1)

        # medcovs = tf.slice(med_cov_grid, [i, 0, 0], [1, -1, -1])
        # tiled_medcovs = tf.tile(medcovs, [n_mc_smps, 1, 1])
        # padded_GPdraws_medcovs = tf.concat([padded_GP_draws, tiled_medcovs], 2)

        # Z = tf.concat([Z, padded_GPdraws_medcovs], 0)
        Z = tf.concat([Z, padded_GP_draws], 0)

        return i + 1, Z

    i = tf.constant(0)
    i, Z = tf.while_loop(cond, body, loop_vars=[i, Z],
                         shape_invariants=[i.get_shape(), tf.TensorShape([None, None, None])])

    return Z


def get_preds(Y, T, X, ind_kf, ind_kt, num_obs_times, num_obs_values,
              num_rnn_grid_times):
    """
    helper function. takes in (padded) raw datas, samples MGP for each observation,
    then feeds it all through the LSTM to get predictions.

    inputs:
        Y: array of observation values (labs/vitals). batchsize x batch_maxlen_y
        T: array of observation times (times during encounter). batchsize x batch_maxlen_t
        ind_kf: indiceste into each row of Y, pointing towards which lab/vital. same size as Y
        ind_kt: indices into each row of Y, pointing towards which time. same size as Y
        num_obs_times: number of times observed for each encounter; how long each row of T really is
        num_obs_values: number of lab values observed per encounter; how long each row of Y really is
        num_rnn_grid_times: length of even spaced RNN grid per encounter

    returns:
        predictions (unnormalized log probabilities) for each MC sample of each obs
    """
    Z = get_GP_samples(Y, T, X, ind_kf, ind_kt, num_obs_times, num_obs_values,
                       num_rnn_grid_times)  # batchsize*num_MC x batch_maxseqlen x num_inputs
    Z.set_shape([None, None, input_dim])  # somehow lost shape info, but need this
    N = tf.shape(T)[0]  # number of observations
    # duplicate each entry of seqlens, to account for multiple MC samples per observation
    seqlen_dupe = tf.reshape(tf.tile(tf.expand_dims(num_rnn_grid_times, 1), [1, n_mc_smps]), [N * n_mc_smps])

    # with tf.variable_scope("",reuse=True):
    outputs, states = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=Z,
                                        dtype=tf.float32,
                                        sequence_length=seqlen_dupe)

    # stacked_rnn_output = tf.reshape(outputs[:, -1], [-1, n_hidden])
    # stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
    # preds = tf.reshape(stacked_outputs, [-1, output])
    # return preds

    # final_outputs=outputs[:,-1]
    final_outputs = states[n_layers - 1][1]
    preds = tf.matmul(final_outputs, out_weights) + out_biases
    # preds=final_outputs
    # preds=Z
    return preds


if __name__ == "__main__":
    seed = 8675309
    rs = np.random.RandomState(seed)  # fixed seed in np

    import matplotlib.pyplot as plt
    import pandas as pd
    from pandas import read_csv

    dataframe = read_csv('energydata_complete.csv', usecols=[1, 2, 3, 4, 5, 6], engine='python')
    dataset1 = dataframe.values
    dataset1 = dataset1.astype('float32')
    dataset1 = dataset1[0:19600, :]


    Real = np.array(dataset1)
    Real[:, 0] = np.add(Real[:, 0], Real[:, 1])
    Real_energy_consumtion = np.delete(Real, 1, axis=1)
    Real_energy_consumtion = Real_energy_consumtion[19100:19300, 0]

    dataset = np.array(dataset1)
    dataset[:, 0] = np.add(dataset[:, 0], dataset[:, 1])
    dataset = np.delete(dataset, 1, axis=1)

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

    for p in range(1,dataset_tr.shape[1]):
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

    # #################
    # ###imupatuion
    # #################

    # import numpy as np
    # a=[0,0,0,3,1,1,5,4,4,6,0,0,0,1,2,5,6,0,4,0,0,7,8,9,0,0,0,0]

    # for n in range(dataset.shape[1]):
    for n in range(1):
        a = dataset[:, n]
        a = np.array(a)
        # for i in range(len(a)):
        p = np.where(a != 0)
        p = np.array(p)
        p = p.reshape(p.shape[1])
        # print(p)
        if p[0] != 0:
            for i in range(p[0]):
                a[i] = a[p[0]]
        # print(a)
        if p[len(p) - 1] != 0:
            for i in range(p[len(p) - 1], len(a)):
                a[i] = a[p[len(p) - 1]]
        # print(a)

        for i in range(len(p) - 1):
            for j in range(p[i] + 1, p[i + 1]):
                a[j] = (a[p[i]] + a[p[i + 1]]) / 2
        # print(a)
        dataset[:, n] = a
    print(dataset)
    # Real_energy_consumtion = dataset[19100:19300, 0]

    # scaling data
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=(0, 1))
    traning_set_scaled = sc.fit_transform(dataset)
    # creating a data structure with 60 steps and 1 output
    x_train = []
    y_train = []
    for i in range(30, 19300):
        # x_train.append(traning_set_scaled[i - 60:i, 0:26])
        x_train.append(traning_set_scaled[i - 30:i, 0:5])
        y_train.append(traning_set_scaled[i, 0])
        # y_train.append(traning_set_scaled[i - 59:i + 1, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    #     print("\tx_train.shape:", x_train.shape)
    #     print(y_train.shape)

    #####
    ##### Setup ground truth and sim some data from a GP
    #####

    # num_encs = 1000  # num_encs=5000
    num_rowdata = x_train.shape[0]
    length_of_any_row_data = x_train.shape[1]
    M = x_train.shape[2]
    # n_covs = 10
    # n_meds = 5

    (num_obs_times, num_obs_values, num_rnn_grid_times, rnn_grid_times, times,
     values, ind_lvs, ind_times) = sim_dataset(num_rowdata, M, trainfrac=0.5)
    # N_tot = len(labels)  # total encounters

    print("\tnum_obs_times:", num_obs_times.shape)
    print("\tnum_obs_values:", num_obs_values.shape)
    print("\tnum_rnn_grid_times:", num_rnn_grid_times.shape)
    print("\trnn_grid_times:", rnn_grid_times.shape)
    print("\ttimes:", times.shape)
    print("\tvalues:", values.shape)
    print("\tind_lvs:", ind_lvs.shape)
    print("\tind_times:", ind_times.shape)

    N_tot = values.shape[0]  # total encounters
    print("\tN_tot:", N_tot)
    #     train_test_perm = rs.permutation(N_tot)
    val_frac = 200 / 19300  # fraction of full data to set aside for testing
    #     te_ind = train_test_perm[:int(val_frac * N_tot)]
    #     tr_ind = train_test_perm[int(val_frac * N_tot):]
    tr_ind = []
    for i in range(19100 - 30):
        tr_ind.append(i)
    te_ind = []
    for i in range(19100 - 30, 19300 - 30):
        te_ind.append(i)
    Nte = len(te_ind);
    Ntr = len(tr_ind)

    # Break everything out into train/test
    # covs_tr = covs[tr_ind, :];
    # covs_te = covs[te_ind, :]
    # labels_tr = labels[tr_ind];
    # labels_te = labels[te_ind]
    target_tr = y_train[tr_ind]
    target_te = y_train[te_ind]
    times_tr = times[tr_ind];
    times_te = times[te_ind]
    values_tr = values[tr_ind];
    values_te = values[te_ind]
    ind_lvs_tr = ind_lvs[tr_ind];
    ind_lvs_te = ind_lvs[te_ind]
    ind_times_tr = ind_times[tr_ind];
    ind_times_te = ind_times[te_ind]
    # meds_on_grid_tr = meds_on_grid[tr_ind];
    # meds_on_grid_te = meds_on_grid[te_ind]
    num_obs_times_tr = num_obs_times[tr_ind];
    num_obs_times_te = num_obs_times[te_ind]
    num_obs_values_tr = num_obs_values[tr_ind];
    num_obs_values_te = num_obs_values[te_ind]
    rnn_grid_times_tr = rnn_grid_times[tr_ind];
    rnn_grid_times_te = rnn_grid_times[te_ind]
    num_rnn_grid_times_tr = num_rnn_grid_times[tr_ind];
    num_rnn_grid_times_te = num_rnn_grid_times[te_ind]

    print("data fully setup!")
    #     sys.stdout.flush()

    #####
    ##### Setup model and graph
    #####

    # Learning Parameters
    learning_rate = 0.001  # NOTE may need to play around with this or decay it
    L2_penalty = 1e-2  # NOTE may need to play around with this some or try additional regularization
    batch_size = 8  # NOTE may want to play around with this
    training_iters = 100  # num epochs=25
    test_freq = Ntr / batch_size  # eval on test set after this many batchess
    # TODO: add dropout regularization

    # Network Parameters
    n_hidden = 30  # hidden layer num of features; assumed same
    n_layers = 3  # number of layers of stacked LSTMs
    n_classes = 2  # binary outcome
    # input_dim = M + n_meds + n_covs  # dimensionality of input sequence.
    input_dim = M  # dimensionality of input sequence.
    n_mc_smps = 35  #####fek konam manzor sampelhay monte carlo ast!!!!!!!!!

    # Create graph
    ops.reset_default_graph()
    #     sess = tf.Session()

    ##### tf Graph - inputs
    output = 1
    # observed values, times, inducing times; padded to longest in the batch
    Y = tf.placeholder("float", [None, None])  # batchsize x batch_maxdata_length
    target = tf.placeholder(tf.float32, [None])
    T = tf.placeholder("float", [None, None])  # batchsize x batch_maxdata_length
    ind_kf = tf.placeholder(tf.int32, [None, None])  # index tasks in Y vector
    ind_kt = tf.placeholder(tf.int32, [None, None])  # index inputs in Y vector
    X = tf.placeholder("float", [None, None])  # grid points. batchsize x batch_maxgridlen
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # med_cov_grid = tf.placeholder("float", [None, None, n_meds + n_covs])  # combine w GP smps to feed into RNN

    # O = tf.placeholder(tf.int32, [None])  # labels. input is NOT as one-hot encoding; convert at each iter
    num_obs_times = tf.placeholder(tf.int32, [None])  # number of observation times per encounter
    num_obs_values = tf.placeholder(tf.int32, [None])  # number of observation values per encounter
    num_rnn_grid_times = tf.placeholder(tf.int32, [None])  # length of each grid to be fed into RNN in batch

    N = tf.shape(Y)[0]

    # also make O one-hot encoding, for the loss function
    # O_dupe_onehot = tf.one_hot(tf.reshape(tf.tile(tf.expand_dims(O, 1), [1, n_mc_smps]), [N * n_mc_smps]), n_classes)
    target_target = tf.reshape(tf.reshape(tf.tile(tf.expand_dims(target, 1), [1, n_mc_smps]), [N * n_mc_smps]),
                               [N * n_mc_smps, 1])

    ##### tf Graph - variables to learn

    ### GP parameters (unconstrained)

    # in fully separable case all labs share same time-covariance
    log_length = tf.Variable(tf.random_normal([1], mean=1, stddev=0.1), name="GP-log-length")
    length = tf.exp(log_length)

    # different noise level of each lab
    log_noises = tf.Variable(tf.random_normal([M], mean=-2, stddev=0.1), name="GP-log-noises")
    noises = tf.exp(log_noises)

    # init cov between labs
    L_f_init = tf.Variable(tf.eye(M), name="GP-Lf")
    Lf = tf.matrix_band_part(L_f_init, -1, 0)
    Kf = tf.matmul(Lf, tf.transpose(Lf))

    ### RNN params

    #     Create network
    #     stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden) for _ in range(n_layers)])

    # Create network
    # stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_hidden) for _ in range(n_layers)])
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    cell2 = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    cell3 = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    cell4 = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True)
    # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.5)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.2)
                                                   , tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=0.2)
                                                   , tf.contrib.rnn.DropoutWrapper(cell3, output_keep_prob=0.2)
                                                   , tf.contrib.rnn.DropoutWrapper(cell3, output_keep_prob=0.2)]
                                               , state_is_tuple=True)

    # Weights at the last layer given deep LSTM output
    # out_weights = tf.Variable(tf.random_normal([n_hidden, n_classes], stddev=0.1), name="Softmax/W")
    out_weights = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.1), name="Softmax/W")
    # out_biases = tf.Variable(tf.random_normal([n_classes], stddev=0.1), name="Softmax/b")
    out_biases = tf.Variable(tf.random_normal([1], stddev=0.1), name="Softmax/b")

    ##### Get predictions and feed into optimization
    preds = get_preds(Y, T, X, ind_kf, ind_kt, num_obs_times, num_obs_values, num_rnn_grid_times)

    # Define optimization problem
    loss_fit = tf.reduce_sum(tf.square(preds - target_target))
    with tf.variable_scope("", reuse=True):
        loss_reg = L2_penalty * tf.reduce_sum(tf.square(out_weights))
        for i in range(n_layers):
            loss_reg = L2_penalty + tf.reduce_sum(
                tf.square(tf.get_variable('rnn/multi_rnn_cell/cell_' + str(i) + '/basic_lstm_cell/kernel')))
    loss = loss_fit + loss_reg
    training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # loss = tf.reduce_sum(tf.square(preds - target_target))
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # training_op = optimizer.minimize(loss)
    ######################################################

    #     init = tf.global_variables_initializer()

    #     epochs = 4
    # Ntr=1656###1840
    # Ntr=1840
    #     sess = tf.Session()
    starts = np.arange(0, Ntr, batch_size)
    ends = np.arange(batch_size, Ntr + 1, batch_size)
    if ends[-1] < Ntr:
        ends = np.append(ends, Ntr)
    num_batches = len(ends)
    #     saver = tf.train.Saver(max_to_keep = None)
    #
    #     total_batches = 0
    #### Main training loop
    T_pad_te, Y_pad_te, ind_kf_pad_te, ind_kt_pad_te, X_pad_te = pad_rawdata(
        times_te, values_te, ind_lvs_te, ind_times_te, rnn_grid_times_te)

    import math
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    ##### Main training loop
    #############################################################################################################
    #     saver = tf.train.Saver(max_to_keep = None)

    total_batches = 0
    with tf.Session() as sess:
        #    init.run()

        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # defaults to saving all variables
        saver.restore(sess, "MGP-miss_50-winmiss=35-4layer_smp35_hiiden30_dropout0.2-5feature((1,2),23,24,25,26)/MGP-181184")

        for i in range(training_iters):
            # epoch_start = time()
            # print("\tepoch:", i)
            # # print("Starting epoch " + "{:d}".format(i))
            # perm = rs.permutation(Ntr)
            # batch = 0
            # for s, e in zip(starts, ends):
            #     batch_start = time()
            #     inds = perm[s:e]
            #     T_pad, Y_pad, ind_kf_pad, ind_kt_pad, X_pad = pad_rawdata(
            #         times_tr[inds], values_tr[inds], ind_lvs_tr[inds], ind_times_tr[inds],
            #         rnn_grid_times_tr[inds])
            #     # print("inds:%s"%len(inds))
            #     feed_dict = {Y: Y_pad, T: T_pad, ind_kf: ind_kf_pad, ind_kt: ind_kt_pad, X: X_pad,
            #                  num_obs_times: num_obs_times_tr[inds],
            #                  num_obs_values: num_obs_values_tr[inds],
            #                  num_rnn_grid_times: num_rnn_grid_times_tr[inds], target: target_tr[inds]}
            #
            #     # F=tf.shape(preds)
            #     # G=tf.shape(target_target)
            #     # print("\tpreds:",sess.run(F,feed_dict))
            #     # print(M)
            #     # print("\tpreds:",sess.run(preds,feed_dict))
            #     # print("\ttarget:",sess.run(target,feed_dict))
            #     # print("\ttarget_target:",sess.run(target_target,feed_dict))
            #     # print("\ttarget_target:",sess.run(G,feed_dict))
            #
            #     # sess.run(training_op, feed_dict={X: x_train[s:e, :num_periods, :inputs], y: y_train[s:e, :output]})
            #     sess.run([loss, training_op], feed_dict)
            #     #                 sys.stdout.flush()
            #     batch += 1;
            #     total_batches += 1
            #     # print(inds)

            if i % 5 == 0:
                tot = total_batches % test_freq
                print("\ttot:", i)

                # saved_path = saver.save(sess, 'checkpoint4444_4layer_smp35_hiiden30_dropout0.2/MGP',
                #                         global_step=global_step)
                print("\tnum_epoch:", i)
                test_t = time()
                feed_dict1 = {Y: Y_pad_te, T: T_pad_te, ind_kf: ind_kf_pad_te, ind_kt: ind_kt_pad_te, X: X_pad_te,
                              num_obs_times: num_obs_times_te,
                              num_obs_values: num_obs_values_te, num_rnn_grid_times: num_rnn_grid_times_te}
                y_pred = sess.run(preds, feed_dict1)
                # print(y_pred)
                y_pred = np.array(y_pred)
                # print("\ty_pred.shape:",y_pred.shape)
                predicted_energy_consumtion = y_pred
                m = np.zeros(((predicted_energy_consumtion.shape[0]), 4))
                V = np.c_[predicted_energy_consumtion, m]
                # print(V.shape)
                predicted_energy_consumtion = sc.inverse_transform(V)
                # print("\tpredicted",predicted_energy_consumtion[:, 0])
                sh = predicted_energy_consumtion.shape[0]
                p = int(sh / n_mc_smps)
                # print(p)
                A = []
                for i in range(0, p):
                    E = 0
                    for j in range(0, n_mc_smps):
                        E = E + predicted_energy_consumtion[j + (i * n_mc_smps), 0]
                    E = E / n_mc_smps
                    A.append(E)
                predicted_energy_consumtion = np.array(A)
                print("\tpredict.shape:", predicted_energy_consumtion.shape)
                #               print("\tpredicted:", predicted_energy_consumtion)
                #         B = np.reshape(target_te, (predicted_energy_consumtion.shape[0], 1))
                #         mm = np.zeros(((predicted_energy_consumtion.shape[0]), 4))
                #         VV = np.c_[B, mm]
                #         # print(VV.shape)
                #         Real_energy_consumtion = sc.inverse_transform(VV)
                Real_energy_consumtion = np.array(Real_energy_consumtion)
                print("\tReal.shape:", Real_energy_consumtion.shape)
                #               print("\tReal:", Real_energy_consumtion)

                #################################

                RMSE = math.sqrt(mean_squared_error(Real_energy_consumtion, predicted_energy_consumtion))
                MAE = mean_absolute_error(Real_energy_consumtion, predicted_energy_consumtion)
                print('Test Score: %.2f RMSE' % (RMSE))
                print('Test Score: %.2f MAE' % (MAE))
                # visualizing the Results
                plt.plot(Real_energy_consumtion, color='red', label='real data')
                plt.plot(predicted_energy_consumtion, color='blue', label='predicted data')
                plt.title('energy_prediction')
                plt.xlabel('Time')
                plt.ylabel('value')
                plt.legend()
                #                 plt.show()
                plt.show()







