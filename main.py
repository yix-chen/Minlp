# import math
# import sys
# import numpy as np
# import matlab
# import matlab.engine
import os
import data.data_generate as dg
import policy.policy_learn as pl
# import svmutil

# engine = matlab.engine.start_matlab()

def main():

    num_train = 10  # 训练集的数目
    num_val = 20
    epoch = 3  # 训练轮数
    #basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_10_4'

    basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_7_2'
    #basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_4_2'
    mattrain_path = basepath + '/train/train.mat'  # 训练数据集的路径
    matval_path = basepath + '/val/val.mat'  # 验证数据集的路径

    '''
    dirs = basepath + '/oracle/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    print("generate oracle...")
    dg.data_generate(mattrain_path, 0, num_train)  # 生成train数据
    '''

    # 已生成 train 10
    

    '''
    dirs = basepath + '/val/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    print("generate validation set...")
    dg.data_generate(matval_path, 1, num_val)  # 生成val数据
    '''

    print('learn policy...')
    pl.prune_learn(mattrain_path, matval_path, epoch, num_train, num_val)
    print('done')


if __name__ == '__main__':
    main()

