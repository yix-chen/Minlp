# -*- coding:utf-8 -*-
import math
import sys
import numpy as np
import matlab
import matlab.engine
import os
import re
import time
from libsvm.svmutil import *
from libsvm.commonutil import *
from libsvm.svm import *

eng = matlab.engine.start_matlab()


#basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_10_4'
#basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_4_2'

basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_7_2'

class TreeNode:
    def __init__(self, x_d, depth):
        self.x_d = x_d  # 确定的x
        self.depth = depth  # 节点深度，根节点为0
        self.p = np.array([])  # 功率松弛解
        self.x_opt = np.array([])  # 松弛最优解
        self.upper = sys.maxsize  # 节点上界
        self.exitflag = sys.maxsize  # 是否有松弛解
        self.fathomed_flag = 0  # 是否满足原始剪枝条件,默认为0即未被剪枝
        self.global_lower_bound = sys.maxsize  # 遍历到节点处的全局下界
        self.status = 0  # 节点标记，默认为-1
        self.parent = None  # 保存父节点
        self.plunge_depth = -1
        self.estobj = sys.maxsize
        self.gapobj = sys.maxsize
        self.leaf = -1  # 默认不是叶节点
        self.solfound = 0
        self.branch = 0
        self.power = 0  # 关于power的问题特征 p_kl_max/p_max_d
        self.channel = 0  # 关于channel的问题特征 log(1+1/(a_kl+b_kl))/R_min_c
        self.sinr = 0
        self.sinr_avgright = 0
        self.sinr_avgleft = 0
        self.sinr_maxright = 0
        self.sinr_maxleft = 0
        # 问题特征

    def setPlungeDepth(self, pdepth):
        # Set PlungeDepth according to number of while-loop iterations
        self.plunge_depth += pdepth


# 调用minlpsolve_sca函数求解松弛问题
# 输入：节点对象
# 输出：x, yita_max, exitflag
def minlpsolve_sca(node):
    print("minlp_solving...")
    x_d = matlab.double(node.x_d.tolist())
    x, p, yita_max, exitflag = eng.minlpsolve_sca(num_user, num_bs,
                                                  gain, power_bs, bandwidth,
                                                  x_d, nargout=4)
    # x, p, yita_max, exitflag
    if exitflag >= 0:
        x = np.array(x).squeeze()
        p = np.array(p).squeeze()

    # print x
    # print p
    return x, p, yita_max, exitflag


# 调用matlab函数求解参数
# 输入: 无
# 输出：sinr, and maybe other features
def para():
    global sinr
    sinr = eng.sinr(gain, power_bs, num_user, num_bs)
    sinr = np.array(sinr).reshape((int(num_user * num_bs), 1))
    #b = np.array(b).reshape((int(K * L), 1))
    #p_max = np.array(p_max).reshape((int(K * L), 1))
    return sinr


# binaryPro:
# branch-and-bound 求解原问题
# mode 0 : 生成Oracle数据(label, feature)
# mode 1 : 生成val数据(label, feature)
def binaryPro(num_user0, num_bs0, bandwidth0, gain0, power_bs0, index, mode):  # index表示问题序号，用于存储文件
    # 转化为全局变量
    global num_user, num_bs, bandwidth, gain, power_bs
    num_user = float(num_user0)
    num_bs = float(num_bs0)
    bandwidth = float(bandwidth0)
    gain = matlab.double(gain0.tolist())
    gain.reshape((num_user0, num_bs0))
    power_bs = matlab.double(power_bs0.tolist())




    sinr = para()

    global globallower  # 全局下界
    global globalx_opt  # 全局下界对应可行解
    global globalnode_opt  # 全局最优点
    global sol_found

    sol_found = 0
    maxdepth = num_user0 * num_bs0
    root_flag = 1

    tolerance = 1.0E-3  # 整数允许误差
    globallower = -sys.maxsize
    depthincrease = 0

    root = TreeNode(np.array([]), 0)  # 根节点
    nodeStack = []  # 树节点堆栈
    nodeList = []  # 保存所有遍历过的节点
    nodeStack.append(root)  # 放入根节点

    start_time = time.time()
    nodenum = 0 # 节点总数
    nodefathom = 0
    while (len(nodeStack) > 0):
        nodenum = nodenum + 1
        node = nodeStack.pop(-1)  # 弹出一个节点
        nodeList.append(node)  # 将节点保存到list中

        print("pop node...")
        print("node.depth:", node.depth)
        print("node.x_d:", node.x_d)
        depthincrease += 1
        node.setPlungeDepth(depthincrease)

        node.x_opt, node.p, node.upper, node.exitflag = minlpsolve_sca(node)
        if root_flag == 1:
            root_bound = node.upper
            root_flag = 0
        node.estobj = root_bound - node.upper
        node.global_lower_bound = globallower  # 保存遍历当前节点时的全局下界

        ind = node.x_d.size
        if ind == num_user0 * num_bs0:  # 叶子结点
            node.leaf = 1
        else:
            x_d_k = math.ceil(ind / num_bs)
            if x_d_k != 0:
                x_d_l = int(ind - (x_d_k - 1) * num_bs)
            else:
                x_d_l = 0
            if node.depth == 0:
                line_ind = 0
                remainder_right = num_bs
                remainder_left = 0

            else:
                line_ind = int((x_d_k - 1) * num_bs + x_d_l)
                remainder_right = int(x_d_k * num_bs) - line_ind
                remainder_left = num_bs - remainder_right - 1

            node.sinr = sinr[line_ind - 1]
            sinr_sumright = 0
            sinr_sumleft = 0

            for i in range(remainder_right):
                sinr_sumright = sinr_sumright + sinr[line_ind+i]
                if sinr[line_ind+i] > node.max_right:
                    node.max_right = sinr[line_ind+i]
            if remainder_right != 0:
                sinr_sumright /= remainder_right
                node.sinr_avgright = sinr_sumright
            else:
                node.sinr_avgright = 0

            for j in range(remainder_left):
                sinr_sumleft = sinr_sumleft + sinr[line_ind-2-i]
                if sinr[line_ind-2-i] > node.max_left:
                    node.max_left = sinr[line_ind-2-i]
            if remainder_left != 0:
                sinr_sumleft /= remainder_left
                node.sinr_avgleft = sinr_sumleft
            else:
                node.sinr_avgleft = 0





            # node.power = K * L * p_max[line_ind] / p_max.sum()
            # node.channel = math.log(1 + 1 / (a[line_ind] + b[line_ind]), 2) / R_min_C

        if node.exitflag < 0:  # 无可行解
            print("Infeasible...")
            node.fathomed_flag = 1  # 被bb剪枝
        elif node.exitflag == 0:  # 未找到最优解
            print("MINLP solver stopped...")
            # 直接对下一个变量进行branch，不剪枝
            ind = node.x_d.size
            node.upper = node.parent.upper
            if ind == num_user0 * num_bs0 or node.upper < globallower:  # 叶子结点或者下界小于上界
                node.fathomed_flag = 1  # 被bb剪枝
                print("Pruning...")
            else:  # 非叶子节点，直接branch
                print("Branching variable index:%d" % ind)
                x_d1 = np.append(node.x_opt[0:ind].copy(), 1)

                # 有一个1后面全补0
                while x_d1.size % num_bs0 != 0:
                    x_d1 = np.append(x_d1, 0)

                x_d0 = np.append(node.x_opt[0:ind].copy(), 0)

                # 生成子节点
                node1 = TreeNode(x_d1, node.depth + 1)
                node0 = TreeNode(x_d0, node.depth + 1)
                node1.parent = node
                node0.parent = node
                node1.branch = 1

                if node.x_opt[ind] > 1 / num_bs:  # 大于1/L，先branch1，即先放入node0，再放入node1
                    nodeStack.append(node0)
                    nodeStack.append(node1)
                else:  # 小于1/L，先branch0，即先放入node1，再放入node0
                    nodeStack.append(node1)
                    nodeStack.append(node0)
        else:  # 找到最优解
            if node.upper < globallower:  # 当前节点上界小于全局下界
                print("Pruning...")
                node.fathomed_flag = 1  # 被bb剪枝
            elif all(
                    ((x - math.floor(x)) < tolerance or (math.ceil(x) - x) < tolerance) for x in node.x_opt):  # 存在整数解
                sol_found = sol_found + 1
                node.x_d = node.x_opt.round()  # 取整

                node.x_opt, node.p, node.upper, node.exitflag = minlpsolve_sca(node)
                node.fathomed_flag = 1  # 被bb剪枝
                print("Found integer solution...")
                if node.upper > globallower:
                    print("Updating global lower bound...")
                    globallower = node.upper
                    globalx_opt = node.x_opt
                    globalnode_opt = node
                    node.global_lower_bound = globallower
                    print("global lower bound:", globallower)
                    print("gloabl optimal solution:")
                    print(node.x_opt)
            else:  # 非整数解
                # 找到第一个非整数解并进行branch
                print("Found non integer solution...")
                ind = [i for i, x in enumerate(node.x_opt) if
                       (x - math.floor(x)) > tolerance and (math.ceil(x) - x) > tolerance][0]

                for j in range(0, ind):  # 取整
                    node.x_opt[j] = node.x_opt[j].round()

                print("Branching variable index:%d" % ind)
                x_d1 = np.append(node.x_opt[0:ind].copy(), 1)

                while x_d1.size % num_bs0 != 0:
                    x_d1 = np.append(x_d1, 0)

                x_d0 = np.append(node.x_opt[0:ind].copy(), 0)

                # 生成子节点
                node1 = TreeNode(x_d1, node.depth + 1)
                node0 = TreeNode(x_d0, node.depth + 1)
                node1.parent = node
                node0.parent = node
                node1.branch = 1

                if node.x_opt[ind] > 1 / num_bs0:  # 大于1/L，先branch1，即先放入node0，再放入node1
                    nodeStack.append(node0)
                    nodeStack.append(node1)
                else:  # 小于1/L，先branch0，即先放入node1，再放入node0
                    nodeStack.append(node1)
                    nodeStack.append(node0)
        node.solfound = sol_found
        node.gapobj = node.upper - globallower
        print('\n')

    total_time = time.time() - start_time
    print('traceback...')
    node = globalnode_opt
    while node is not None:
        node.status = 1
        print(node.x_d, '-->')
        node = node.parent

    label = np.array([])
    feature = np.array([])

    if mode == 0:
        # oracle模式
        optPath = basepath + '/oracle/problem' + str(index) + '_optimal.txt'
        if os.path.exists(optPath):
            os.remove(optPath)
        optFile = open(optPath, mode='a+')
        resPath = basepath + '/oracle/problem' + str(index) + '_result.txt'
        if os.path.exists(resPath):
            os.remove(resPath)
        resFile = open(resPath, mode='a+')
        restPath = basepath + '/oracle/problem_result.txt'
        if index == 1:
            if os.path.exists(restPath):
                os.remove(restPath)
        restFile = open(restPath, mode='a+')

        while len(nodeList) > 0:
            node = nodeList.pop(0)
            if node.status == 1:
                optFile.writelines([str(node.x_d), '\n'])
            resFile.writelines(
                [str(node.x_d), '\t', str(node.x_opt).replace("\n", " "), '\t', str(node.p).replace("\n", " "),
                 '\t',
                 str(node.upper), '\t', str(node.exitflag), '\n'])
        # 全局特征？
        restFile.writelines(
            [str(index), '\tgloballower:\t', str(globallower), '\ttime:\t', str(total_time), '\tnodenum:\t',
             str(nodenum), '\tx_opt:\t', str(globalx_opt).replace("\n", " "), '\tp_opt:\t',
             str(globalnode_opt.p).replace("\n", " "), '\n'])
        optFile.close()
        resFile.close()
        restFile.close()

    else:
        # 生成val数据
        flag = 0

        optPath = basepath + '/val/problem' + str(index) + '_optimal.txt'
        if os.path.exists(optPath):
            os.remove(optPath)
        optFile = open(optPath, mode='a+')
        resPath = basepath + '/val/problem' + str(index) + '_result.txt'
        if os.path.exists(resPath):
            os.remove(resPath)
        resFile = open(resPath, mode='a+')
        restPath = basepath + '/val/problem_result.txt'
        if index == 1:
            if os.path.exists(restPath):
                os.remove(restPath)
        restFile = open(restPath, mode='a+')

        while len(nodeList) > 0:
            node = nodeList.pop(0)
            if node.exitflag != 0 and node.fathomed_flag != 1:
                feature_temp = np.array(
                    [node.depth / maxdepth, node.upper / root_bound, node.global_lower_bound / root_bound,
                     node.plunge_depth / maxdepth,
                     node.estobj / root_bound, node.leaf, node.solfound / maxdepth, node.gapobj / root_bound,
                     node.branch, node.sinr])  # node.power, node.channel
                if node.status == 1:
                    label_temp = np.array([1])
                    if flag == 0:
                        flag = 1
                        label = label_temp
                        feature = feature_temp
                    else:
                        label = np.vstack((label, label_temp))
                        feature = np.vstack((feature, feature_temp))
                else:
                    label_temp = np.array([0])
                    if flag == 0:
                        flag = 1
                        label = label_temp
                        feature = feature_temp
                    else:
                        label = np.vstack((label, label_temp))
                        feature = np.vstack((feature, feature_temp))
            if node.status == 1:
                optFile.writelines([str(node.x_d), '\n'])
            resFile.writelines(
                [str(node.x_d), '\t', str(node.x_opt).replace("\n", " "), '\t', str(node.p).replace("\n", " "),
                 '\t',
                 str(node.upper), '\t', str(node.exitflag), '\n'])
        restFile.writelines(
            [str(index), '\tgloballower:\t', str(globallower), '\ttime:\t', str(total_time), '\tnodenum:\t',
             str(nodenum), '\tx_opt:\t', str(globalx_opt).replace("\n", " "), '\tp_opt:\t',
             str(globalnode_opt.p).replace("\n", " "), '\n'])
        optFile.close()
        resFile.close()
        restFile.close()

    return globalx_opt, globallower, label, feature


# binaryPro_oracle
# branch-and-bound + Oracle
# 1. 已解过的点无需再解，未解过的点写入resFile
# 2. exitflag == 0，直接branch
# 3. 找到最优解，存在整数解，剪枝
# 4. 找到最优解，不存在整数解，由Oracle决定是否剪枝
def binaryPro_oracle(num_user0, num_bs0, bandwidth0, gain0, power_bs0, index):  # index表示问题序号，用于搜索文件
    # 转化为全局变量
    global num_user, num_bs, bandwidth, gain, power_bs
    num_user = float(num_user0)
    num_bs = float(num_bs0)
    bandwidth = float(bandwidth0)
    gain = matlab.double(gain0.tolist())
    gain.reshape((num_user0, num_bs0))
    power_bs = matlab.double(power_bs0.tolist())

    sinr = para()

    global globallower  # 全局下界
    global globalx_opt  # 全局下界对应可行解
    global globalnode_opt  # 全局最优点
    global sol_found

    sol_found = 0

    maxdepth = num_user0 * num_bs0
    root_flag = 1
    feature_flag = 1

    tolerance = 1.0E-3  # 整数允许误差
    globallower = -sys.maxsize
    depthincrease = 0

    root = TreeNode(np.array([]), 0)  # 根节点
    nodeStack = []  # 树节点堆栈
    nodeList = []  # 保存所有遍历过的节点
    nodeStack.append(root)  # 放入根节点

    while (len(nodeStack) > 0):
        node = nodeStack.pop(-1)  # 弹出一个节点
        nodeList.append(node)  # 将节点保存到list中

        print("pop node...")
        print("node.depth:", node.depth)
        print("node.x_d:", node.x_d)
        depthincrease += 1
        node.setPlungeDepth(depthincrease)

        resPath = basepath + '/oracle/problem' + str(index) + '_result.txt'
        resFile = open(resPath, mode='r')
        solved = 0  # 是否被解过
        line_temp = resFile.readline()
        while line_temp:
            if str(node.x_d) in line_temp:
                solved = 1
                break
            else:
                line_temp = resFile.readline()
        resFile.close()
        if solved == 0:  # 没有解过
            resPath = basepath + '/oracle/problem' + str(index) + '_result.txt'
            resFile = open(resPath, mode='a+')
            node.x_opt, node.p, node.upper, node.exitflag = minlpsolve_sca(node)  # 求解松弛问题
            resFile.writelines(
                [str(node.x_d), '\t', str(node.x_opt).replace("\n", " "), '\t', str(node.p).replace("\n", " "),
                 '\t',
                 str(node.upper), '\t', str(node.exitflag), '\n'])
            resFile.close()
        else:  # 已经解过
            temp = (line_temp.strip()).split('\t')
            x_d2 = temp[1] # ?
            temp_a = re.findall(r'[[](.*)[]]', temp[1].strip())
            node.x_opt = np.array(list(map(eval, temp_a[0].split())))
            temp_a = re.findall(r'[[](.*)[]]', temp[2].strip())
            node.p = np.array(list(map(eval, temp_a[0].split())))
            node.upper = float(temp[3])
            node.exitflag = float(temp[4])

        if root_flag == 1:
            root_bound = node.upper
            root_flag = 0
        node.estobj = root_bound - node.upper
        node.global_lower_bound = globallower  # 保存遍历当前节点时的全局下界

        ind = node.x_d.size
        if ind == num_user0 * num_bs0:  # 叶子结点
            node.leaf = 1
        else:
            x_d_k = math.ceil(ind / num_bs)
            if x_d_k != 0:
                x_d_l = int(ind - (x_d_k - 1) * num_bs)
            else:
                x_d_l = 0
            if node.depth == 0:
                line_ind = 1
            else:
                line_ind = int((x_d_k - 1) * num_bs + x_d_l)

            node.sinr = sinr[line_ind]
            # node.power = K * L * p_max[line_ind] / p_max.sum()
            # node.channel = math.log(1 + 1 / (a[line_ind] + b[line_ind]), 2) / R_min_C

        optPath = basepath + '/oracle/problem' + str(index) + '_optimal.txt'
        optFile = open(optPath, mode='r')
        line_temp = optFile.readline()
        while line_temp:
            if str(node.x_d) in line_temp:
                node.status = 1
                break
            else:
                line_temp = optFile.readline()
        optFile.close()

        if node.exitflag < 0:  # 无可行解
            print("Infeasible...")
            node.fathomed_flag = 1  # 被bb剪枝
        elif node.exitflag == 0:  # 未找到最优解
            # 未找到最优解
            print("MINLP solver stopped...")
            # 直接对下一个变量进行branch，不剪枝
            ind = node.x_d.size
            node.upper = node.parent.upper
            if ind == num_user0 * num_bs0 or node.upper < globallower:  # 叶子结点或者下界小于上界
                node.fathomed_flag = 1  # 被bb剪枝
                print("Pruning...")
            else:  # 非叶子节点，直接branch
                print("Branching variable index:%d" % ind)
                x_d1 = np.append(node.x_opt[0:ind].copy(), 1)
                while x_d1.size % num_bs0 != 0:
                    x_d1 = np.append(x_d1, 0)
                x_d0 = np.append(node.x_opt[0:ind].copy(), 0)

                # 生成子节点
                node1 = TreeNode(x_d1, node.depth + 1)
                node0 = TreeNode(x_d0, node.depth + 1)
                node1.parent = node
                node0.parent = node
                node1.branch = 1

                if node.x_opt[ind] > 1 / num_bs:  # 大于1/L，先branch1，即先放入node0，再放入node1
                    nodeStack.append(node0)
                    nodeStack.append(node1)
                else:  # 小于1/L，先branch0，即先放入node1，再放入node0
                    nodeStack.append(node1)
                    nodeStack.append(node0)
        else:
            if node.upper < globallower:  # 当前节点上界小于全局下界
                print("Pruning...")
                node.fathomed_flag = 1  # 被bb剪枝
            elif all(
                    ((x - math.floor(x)) < tolerance or (math.ceil(x) - x) < tolerance) for x in node.x_opt):  # 存在整数解
                sol_found = sol_found + 1

                node.x_d = node.x_opt.round()  # 取整
                resFile = open(resPath, mode='r')
                solved = 0  # 是否被解过
                line_temp = resFile.readline()
                while line_temp:
                    if str(node.x_d) in line_temp:
                        solved = 1
                        break
                    else:
                        line_temp = resFile.readline()
                resFile.close()
                if solved == 0:  # 没有解过
                    resPath = basepath + '/oracle/problem' + str(index) + '_result.txt'
                    resFile = open(resPath, mode='a+')
                    node.x_opt, node.p, node.upper, node.exitflag = minlpsolve_sca(node)  # 求解松弛问题
                    resFile.writelines([str(node.x_d), '\t', str(node.x_opt).replace("\n", " "), '\t',
                                        str(node.p).replace("\n", " "), '\t',
                                        str(node.upper), '\t', str(node.exitflag), '\n'])
                    resFile.close()
                else:  # 已经解过
                    temp = (line_temp.strip()).split('\t')
                    temp_a = re.findall(r'[[](.*)[]]', temp[1].strip())
                    node.x_opt = np.array(list(map(eval, temp_a[0].split())))
                    temp_a = re.findall(r'[[](.*)[]]', temp[2].strip())
                    node.p = np.array(list(map(eval, temp_a[0].split())))
                    node.upper = float(temp[3])
                    node.exitflag = float(temp[4])

                node.fathomed_flag = 1  # 被bb剪枝
                node.x_opt = node.x_opt.round()  # 取整
                print("Found integer solution...")
                if node.upper > globallower:
                    print("Updating gloabel lower bound...")
                    globallower = node.upper
                    globalx_opt = node.x_opt
                    globalnode_opt = node
                    node.global_lower_bound = globallower
                    print("global lower bound:", globallower)
                    print("gloabl optimal solution:")
                    print(node.x_opt)

            node.solfound = sol_found
            node.gapobj = node.upper - globallower
            feature_tmp = np.array(
                [node.depth / maxdepth, node.upper / root_bound, node.global_lower_bound / root_bound,
                 node.plunge_depth / maxdepth,
                 node.solfound / maxdepth, node.branch, node.sinr])

            # node.power, node.channel])

            # feature_tmp = np.array([node.depth/maxdepth,node.upper/root_bound,node.global_lower_bound/root_bound,node.plunge_depth/maxdepth,
            #            node.solfound/maxdepth,node.branch])
            label_tmp = np.array([node.status])

            if node.fathomed_flag != 1:
                # 权重
                wePath = basepath + '/weight.txt'
                weFile = open(wePath, mode='a+')
                weFile.writelines([str(5 * math.exp(-2.68 * node.depth / maxdepth)), '\n'])
                weFile.close()

                if node.status == 1:
                    # 写入不剪枝
                    if feature_flag == 1:
                        feature = feature_tmp
                        label = label_tmp
                        feature_flag = 0
                    else:
                        label = np.hstack((label, label_tmp))
                        feature = np.vstack((feature, feature_tmp))
                    # branch
                    print("Model decide branching:...")
                    ind = [i for i, x in enumerate(node.x_opt) if
                           (x - math.floor(x)) > tolerance and (math.ceil(x) - x) > tolerance][0]

                    for j in range(0, ind):  # 取整
                        node.x_opt[j] = node.x_opt[j].round()

                    print("Branching variable index:%d" % ind)
                    x_d1 = np.append(node.x_opt[0:ind].copy(), 1)
                    while x_d1.size % num_bs0 != 0:
                        x_d1 = np.append(x_d1, 0)
                    x_d0 = np.append(node.x_opt[0:ind].copy(), 0)

                    # 生成子节点
                    node1 = TreeNode(x_d1, node.depth + 1)
                    node0 = TreeNode(x_d0, node.depth + 1)
                    node1.parent = node
                    node0.parent = node
                    node1.branch = 1

                    if node.x_opt[ind] > 1 / num_bs:  # 大于1/L，先branch1，即先放入node0，再放入node1
                        nodeStack.append(node0)
                        nodeStack.append(node1)
                    else:  # 小于1/L，先branch0，即先放入node1，再放入node0
                        nodeStack.append(node1)
                        nodeStack.append(node0)
                else:
                    # Model决定剪枝
                    if feature_flag == 1:
                        feature = feature_tmp
                        label = label_tmp
                        feature_flag = 0
                    else:
                        label = np.hstack((label, label_tmp))
                        feature = np.vstack((feature, feature_tmp))

        print('\n')

    return label, feature

# binaryPro_policy
# branch-and-bound + svmmodel
###################################################
# policy的while中的内容
# pop出第一个点p，先fmincon0运行。如果是optimal，则写入特征
# 和branch，如果不是optimal并且exitflag不等于0，则写入特征
# 和prune。如果找到可行解，更新全局界。再判断p没有被bb剪枝
# 并且新的model也判断不剪枝，branch出新的点。并且放入队列中。
def binaryPro_policy(num_user0, num_bs0, bandwidth0, gain0, power_bs0, index, model, scaler):
    # 转化为全局变量
    global num_user, num_bs, bandwidth, gain, power_bs
    num_user = float(num_user0)
    num_bs = float(num_bs0)
    bandwidth = float(bandwidth0)
    gain = matlab.double(gain0.tolist())
    gain.reshape((num_user0, num_bs0))
    power_bs = matlab.double(power_bs0.tolist())

    sinr = para()

    global globallower  # 全局下界
    global globalx_opt  # 全局下界对应可行解
    global globalnode_opt  # 全局最优点
    global sol_found

    sol_found = 0

    maxdepth = num_user * num_bs
    root_flag = 1
    feature_flag = 1

    tolerance = 1.0E-3  # 整数允许误差
    globallower = -sys.maxsize
    depthincrease = 0

    root = TreeNode(np.array([]), 0)  # 根节点
    nodeStack = []  # 树节点堆栈
    nodeList = []  # 保存所有遍历过的节点
    nodeStack.append(root)  # 放入根节点

    while (len(nodeStack) > 0):
        node = nodeStack.pop(-1)  # 弹出一个节点
        nodeList.append(node)  # 将节点保存到list中

        print("pop node...")
        print("node.depth:", node.depth)
        print("node.x_d:", node.x_d)
        depthincrease += 1
        node.setPlungeDepth(depthincrease)

        resPath = basepath + '/oracle/problem' + str(index) + '_result.txt'
        resFile = open(resPath, mode='r')
        solved = 0  # 是否被解过
        line_temp = resFile.readline()
        while line_temp:
            if str(node.x_d) in line_temp:
                solved = 1
                break
            else:
                line_temp = resFile.readline()
        resFile.close()
        if solved == 0:  # 没有解过
            resPath = basepath + '/oracle/problem' + str(index) + '_result.txt'
            resFile = open(resPath, mode='a+')
            node.x_opt, node.p, node.upper, node.exitflag = minlpsolve_sca(node)  # 求解松弛问题
            resFile.writelines(
                [str(node.x_d), '\t', str(node.x_opt).replace("\n", " "), '\t', str(node.p).replace("\n", " "),
                 '\t',
                 str(node.upper), '\t', str(node.exitflag), '\n'])
            resFile.close()
        else:  # 已经解过
            temp = (line_temp.strip()).split('\t')
            temp_a = re.findall(r'[[](.*)[]]', temp[1].strip())
            node.x_opt = np.array(list(map(eval, temp_a[0].split())))
            temp_a = re.findall(r'[[](.*)[]]', temp[2].strip())
            node.p = np.array(list(map(eval, temp_a[0].split())))
            node.upper = float(temp[3])
            node.exitflag = float(temp[4])

        if root_flag == 1:
            root_bound = node.upper
            root_flag = 0
        node.estobj = root_bound - node.upper
        node.global_lower_bound = globallower  # 保存遍历当前节点时的全局下界

        ind = node.x_d.size
        if ind == num_user0 * num_bs0:  # 叶子结点
            node.leaf = 1
        else:
            x_d_k = math.ceil(ind / num_bs)
            if x_d_k != 0:
                x_d_l = int(ind - (x_d_k - 1) * num_bs)
            else:
                x_d_l = 0
            if node.depth == 0:
                line_ind = 1
            else:
                line_ind = int((x_d_k - 1) * num_bs + x_d_l)

            node.sinr = sinr[line_ind]
            # node.power = K * L * p_max[line_ind] / p_max.sum()
            # node.channel = math.log(1 + 1 / (a[line_ind] + b[line_ind]), 2) / R_min_C

        optPath = basepath + '/oracle/problem' + str(index) + '_optimal.txt'
        optFile = open(optPath, mode='r')
        line_temp = optFile.readline()
        while line_temp:
            if str(node.x_d) in line_temp:
                node.status = 1
                break
            else:
                line_temp = optFile.readline()
        optFile.close()

        if node.exitflag < 0:  # 无可行解
            print("Infeasible...")
            node.fathomed_flag = 1  # 被bb剪枝
        elif node.exitflag == 0:  # 未找到最优解
            # 未找到最优解
            print("MINLP solver stopped...")
            # 直接对下一个变量进行branch，不剪枝
            ind = node.x_d.size
            node.upper = node.parent.upper
            if ind == num_user0 * num_bs0 or node.upper < globallower:  # 叶子结点或者下界小于上界
                node.fathomed_flag = 1  # 被bb剪枝
                print("Pruning...")
            else:  # 非叶子节点，直接branch
                print("Branching variable index:%d" % ind)
                x_d1 = np.append(node.x_opt[0:ind].copy(), 1)
                while x_d1.size % num_bs0 != 0:
                    x_d1 = np.append(x_d1, 0)
                x_d0 = np.append(node.x_opt[0:ind].copy(), 0)

                # 生成子节点
                node1 = TreeNode(x_d1, node.depth + 1)
                node0 = TreeNode(x_d0, node.depth + 1)
                node1.parent = node
                node0.parent = node
                node1.branch = 1

                if node.x_opt[ind] > 1 / num_bs:  # 大于1/L，先branch1，即先放入node0，再放入node1
                    nodeStack.append(node0)
                    nodeStack.append(node1)
                else:  # 小于1/L，先branch0，即先放入node1，再放入node0
                    nodeStack.append(node1)
                    nodeStack.append(node0)
        else:
            if node.upper < globallower:  # 当前节点上界小于全局下界
                print("Pruning...")
                node.fathomed_flag = 1  # 被bb剪枝
            elif all(
                    ((x - math.floor(x)) < tolerance or (math.ceil(x) - x) < tolerance) for x in node.x_opt):  # 存在整数解
                sol_found = sol_found + 1

                node.x_d = node.x_opt.round()  # 取整
                resFile = open(resPath, mode='r')
                solved = 0  # 是否被解过
                line_temp = resFile.readline()
                while line_temp:
                    if str(node.x_d) in line_temp:
                        solved = 1
                        break
                    else:
                        line_temp = resFile.readline()
                resFile.close()
                if solved == 0:  # 没有解过
                    resPath = basepath + '/oracle/problem' + str(index) + '_result.txt'
                    resFile = open(resPath, mode='a+')
                    node.x_opt, node.p, node.upper, node.exitflag = minlpsolve_sca(node)  # 求解松弛问题
                    resFile.writelines([str(node.x_d), '\t', str(node.x_opt).replace("\n", " "), '\t',
                                        str(node.p).replace("\n", " "), '\t',
                                        str(node.upper), '\t', str(node.exitflag), '\n'])
                    resFile.close()
                else:  # 已经解过
                    temp = (line_temp.strip()).split('\t')
                    temp_a = re.findall(r'[[](.*)[]]', temp[1].strip())
                    node.x_opt = np.array(list(map(eval, temp_a[0].split())))
                    temp_a = re.findall(r'[[](.*)[]]', temp[2].strip())
                    node.p = np.array(list(map(eval, temp_a[0].split())))
                    node.upper = float(temp[3])
                    node.exitflag = float(temp[4])

                node.fathomed_flag = 1  # 被bb剪枝
                node.x_opt = node.x_opt.round()  # 取整
                print("Found integer solution...")
                if node.upper > globallower:
                    print("Updating gloabel lower bound...")
                    globallower = node.upper
                    globalx_opt = node.x_opt
                    globalnode_opt = node
                    node.global_lower_bound = globallower
                    print("global lower bound:", globallower)
                    print("gloabl optimal solution:")
                    print(node.x_opt)

            node.solfound = sol_found
            node.gapobj = node.upper - globallower
            feature_tmp = np.array(
                [node.depth / maxdepth, node.upper / root_bound, node.global_lower_bound / root_bound,
                 node.plunge_depth / maxdepth,
                 node.solfound / maxdepth, node.branch, node.sinr])
            # node.power, node.channel])
            # feature_tmp =  np.array([node.depth/maxdepth,node.upper/root_bound,node.global_lower_bound/root_bound,node.plunge_depth/maxdepth,
            #            node.solfound/maxdepth,node.branch])
            label_tmp = np.array([node.status])

            if node.fathomed_flag != 1:
                wePath = basepath + '/weight.txt'
                weFile = open(wePath, mode='a+')
                weFile.writelines([str(5 * math.exp(-2.68 * node.depth / maxdepth)), '\n'])
                weFile.close()

                if feature_flag == 1:
                    feature = feature_tmp
                    label = label_tmp
                    feature_flag = 0
                else:
                    label = np.hstack((label, label_tmp))
                    feature = np.vstack((feature, feature_tmp))

                # model = svm_load_model('model_'+ str(model_index)+'.model')
                tmpPath = basepath + '/tmp.txt'
                tmpFile = open(tmpPath, mode='w')
                # feature_check = np.array([node.depth/maxdepth,node.upper/root_bound,node.global_lower_bound/root_bound,node.plunge_depth/maxdepth,
                #        node.estobj/root_bound,node.leaf,node.solfound/maxdepth,node.gapobj/root_bound,node.branch,node.power,node.channel])
                feature_check = np.array(
                    [node.depth / maxdepth, node.upper / root_bound, node.global_lower_bound / root_bound,
                     node.plunge_depth / maxdepth,
                     node.solfound / maxdepth, node.branch, node.sinr])
                     #node.power, node.channel])
                # feature_check = np.array([node.depth/maxdepth,node.upper/root_bound,node.global_lower_bound/root_bound,node.plunge_depth/maxdepth,
                #        node.solfound/maxdepth,node.branch])
                feature_scaled = scaler.transform(feature_check.reshape(1, -1)).reshape(-1)
                tmpFile.writelines([str(node.status)])
                for j in range(len(feature_scaled)):
                    tmpFile.writelines(['\t', str(j), ':', str(feature_scaled[j])])
                tmpFile.writelines(['\n'])
                tmpFile.close()
                # 用model判断
                ############
                modelP, modelF = svm_read_problem(tmpPath)  # labels, data
                p_label, p_acc, p_val = svm_predict(modelP, modelF, model)
                print("Model decision: %d" % p_label[0])
                ############

                if p_label[0] == 1:
                    # branch
                    print("Model decide branching:...")
                    ind = [i for i, x in enumerate(node.x_opt) if
                           (x - math.floor(x)) > tolerance and (math.ceil(x) - x) > tolerance][0]

                    for j in range(0, ind):  # 取整
                        node.x_opt[j] = node.x_opt[j].round()

                    print("Branching variable index:%d" % ind)
                    x_d1 = np.append(node.x_opt[0:ind].copy(), 1)
                    while x_d1.size % num_bs0 != 0:
                        x_d1 = np.append(x_d1, 0)
                    x_d0 = np.append(node.x_opt[0:ind].copy(), 0)

                    # 生成子节点
                    node1 = TreeNode(x_d1, node.depth + 1)
                    node0 = TreeNode(x_d0, node.depth + 1)
                    node1.parent = node
                    node0.parent = node
                    node1.branch = 1

                    if node.x_opt[ind] > 1 / num_bs:  # 大于1/L，先branch1，即先放入node0，再放入node1
                        nodeStack.append(node0)
                        nodeStack.append(node1)
                    else:  # 小于1/L，先branch0，即先放入node1，再放入node0
                        nodeStack.append(node1)
                        nodeStack.append(node0)

        print('\n')

    return label, feature

