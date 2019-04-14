# -*- coding:utf-8 -*-
# imitation learning
import scipy.io as sio
import numpy as np
from libsvm.svmutil import *
from libsvm.commonutil import *
from libsvm.svm import *
import bb.utils as bb
import policy.policy_run as pr
from sklearn import preprocessing
import os



#basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_10_4'
#basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_4_2'

basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_4_2_avgleft'


def prune_learn(matfn, matva, Nround, Nprob, Nval):
    # Nround：循环的轮数
    # Nprob：问题的数目
    # C：svm的参数

    data = sio.loadmat(matfn)
    num_user = data['num_user'][0, 0]
    num_bs = data['num_bs'][0, 0]
    num_instance = data['num_instance'][0, 0]
    bandwidth = data['bandw'][0, 0]
    gain = data['Gain_UB']
    power_bs = data['power_Vector']

    data1 = sio.loadmat(matva)
    num_user1 = data1['num_user'][0, 0]
    num_bs1 = data1['num_bs'][0, 0]
    num_instance1 = data1['num_instance'][0, 0]
    bandwidth1 = data1['bandw'][0, 0]
    gain1 = data1['Gain_UB']
    power_bs1 = data1['power_Vector']

    # 生成训练集文件
    train_path = basepath + '/train' + '.txt'
    if os.path.exists(train_path):
        os.remove(train_path)

    # 生成权重文件
    wePath = basepath + '/weight.txt'
    if os.path.exists(wePath):
        os.remove(wePath)

    # 生成准确率记录文件
    gap_path = basepath + '/gap.txt'

    # 生成模型存储路径
    dirs = basepath + '/model'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # Nprob = H_CD.shape[1]

    # imitation learning
    for r in range(Nround):
        print("round #%d..." % (r + 1))
        for i in range(Nprob):
            print("problem #%d..." % (i + 1))
            gain_i = gain[:, :, i].reshape((num_user, num_bs))
            power_bs_i = power_bs[i, :]

            index = i + 1
            if r == 0:
                # 用oracle控制并写入到train_file中
                label_train_tmp, feature_train_tmp = bb.binaryPro_oracle(num_user, num_bs, bandwidth,
                                                                         gain_i, power_bs_i, index)
            else:
                # 用当前policy控制并写入到train_file中
                label_train_tmp, feature_train_tmp = bb.binaryPro_policy(num_user, num_bs, bandwidth,
                                                                         gain_i, power_bs_i, index,
                                                                         m, scaler)
            if r == 0 and i == 0:
                label_train = label_train_tmp
                feature_train = feature_train_tmp
            else:
                label_train = np.hstack((label_train, label_train_tmp))
                feature_train = np.vstack((feature_train, feature_train_tmp))

        weFile = open(wePath, mode='r')
        sourceInLine = weFile.readlines()
        W = []
        W_temp = []
        for line in sourceInLine:
            temp = line.strip('\n')
            W_temp.append(temp)
        W = list(map(eval, W_temp))
        weFile.close()

        feature_scaled = preprocessing.scale(feature_train)
        scaler = preprocessing.StandardScaler().fit(feature_train)

        train_file = open(train_path, mode='w+')
        train_num = feature_scaled.shape[0]
        for i in range(train_num):
            train_file.writelines([str(label_train[i])])
            for j in range(len(feature_train[i, :])):
                train_file.writelines(['\t', str(j), ':', str(feature_scaled[i, j])])
            train_file.writelines(['\n'])
        train_file.close()

        num1 = 0
        num2 = 0
        train_file = open(train_path, mode='r')
        line = train_file.readline()
        while line != "":
            s = line[:1]
            if s == str(1):
                num1 = num1 + 1
            else:
                num2 = num2 + 1
            line = train_file.readline()
        train_file.close()

        y_, x = svm_read_problem(train_path)
        # W[:] = [i * 1.25 for i in W]
        m = svm_train(W, y_, x, '-c 0.25 -w1 6 -w0 12') # libsvm-weight

        print('test for round %d:' % (r + 1))

        svm_save_model(basepath + '/model/model_' + str(r + 1) + '.model', m)
        gap_file = open(gap_path, mode='a+')
        gap_file.writelines(['round:\t', str(r + 1), '\n'])
        gap_file.close()
        gap_sum = 0
        speed_sum = 0
        acc1_sum = 0
        acc0_sum = 0
        num1_sum = 0
        num0_sum = 0
        for i in range(0, Nval):
            print("problem #%d..." % (i + 1))
            gain1_i = gain1[:, :, i].reshape((num_user1, num_bs1))
            power_bs1_i = power_bs1[i, :]
            index = i + 1
            ogap, speed, acc1, acc0, num1, num0 = pr.policyRun(num_user1, num_bs1, bandwidth1,
                                                               gain1_i, power_bs1_i, index, m, scaler)
            gap_file = open(gap_path, mode='a+')
            gap_file.writelines(['problem:\t', str(index), '\tOgap:\t', str(ogap), '\n'])
            gap_file.close()
            gap_sum = gap_sum + ogap
            speed_sum = speed_sum + speed
            acc1_sum = acc1_sum + acc1
            acc0_sum = acc0_sum + acc0
            num1_sum = num1_sum + num1
            num0_sum = num0_sum + num0
        gap_file = open(gap_path, mode='a+')
        gap_file.writelines(
            ['average gap:\t', str(gap_sum / Nval), '\taverage speed:\t', str(speed_sum / Nval), '\tacc1:\t',
             str(acc1_sum / num1_sum), '\tacc0:\t', str(acc0_sum / num0_sum), '\n'])
        gap_file.close()
