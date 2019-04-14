# -*- coding:utf-8 -*-
# 用于生成数据对应的txt文件
import scipy.io as sio
import numpy as np
import os
import bb.utils as bb

#basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_10_4/'
#basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_4_2/'

basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_4_2_avgleft/'
def data_generate(matfn, mode, num_total):
    data = sio.loadmat(matfn)
    num_user = data['num_user'][0, 0]
    num_bs = data['num_bs'][0, 0]
    num_instance = data['num_instance'][0, 0]
    bandwidth = data['bandw'][0, 0]
    gain = data['Gain_UB']
    power_bs = data['power_Vector']

    #for i in range(num_total):
    for i in (8,9):
        print("solving problem #%d..." % (i + 1))

        gain_i = gain[:, :, i].reshape((num_user, num_bs))
        power_bs_i = power_bs[i, :].reshape((-1, num_bs))

        # 调用branch-and-bound解决第i个instance并存储结果
        index = i + 1

        x, yita_max, label_temp, feature_temp = bb.binaryPro(num_user, num_bs, bandwidth, gain_i, power_bs_i, index, mode)

        if mode == 1 and len(label_temp) != 0:
            if i == 0:
                label = label_temp
                feature = feature_temp
                #label = np.loadtxt(basepath+'val/label_val.txt')
                #feature = np.loadtxt(basepath+'val/feature_val.txt')
                #label = label.reshape((label.shape[0], -1))

                #label = np.vstack((label, label_temp))
                #feature = np.vstack((feature, feature_temp))
            else:
                label = np.vstack((label, label_temp))
                feature = np.vstack((feature, feature_temp))
        else:
            label = np.array([])
            feature = np.array([])

        if mode == 1:
            labelPath = basepath + 'val/label_val.txt'
            featurePath = basepath + 'val/feature_val.txt'

            if os.path.exists(labelPath):
                os.remove(labelPath)
            if os.path.exists(featurePath):
                os.remove(featurePath)
            np.savetxt(featurePath, feature)
            np.savetxt(labelPath, label)

        print("problem #%d solved!" % (i + 1))
        print("final solution:", x)
        print("optimal value:", yita_max)
    return label, feature







