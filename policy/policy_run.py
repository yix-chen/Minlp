import math
import numpy as np
import matlab
import matlab.engine
from libsvm.svmutil import *
from libsvm.commonutil import *
from libsvm.svm import *
import re
import sys
from bb.utils import TreeNode
from bb.utils import para
from bb.utils import minlpsolve_sca

eng = matlab.engine.start_matlab()
#basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_10_4'

#basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_4_2'

basepath = '/Users/chyx/Study/JuniorSpringSummer/SrtpMinlp/simulation/data_7_2'
def policyRun(num_user0, num_bs0, bandwidth0, gain0, power_bs0, index, model, scaler):
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
    acc1 = 0
    acc0 = 0
    num1 = 0
    num0 = 0

    maxdepth = num_user * num_bs
    root_flag = 1

    tolerance = 1.0E-3  # 整数允许误差
    globallower = -sys.maxsize
    depthincrease = 0

    root = TreeNode(np.array([]), 0)  # 根节点
    nodeStack = []  # 树节点堆栈
    nodeList = []  # 保存所有遍历过的节点
    nodeStack.append(root)  # 放入根节点

    # start_time = time.time()
    nodenum = 0
    while (len(nodeStack) > 0):
        nodenum = nodenum + 1
        node = nodeStack.pop(-1)  # 弹出一个节点
        nodeList.append(node)  # 将节点保存到list中

        print("pop node...")
        print("node.depth:", node.depth)
        print("node.x_d:", node.x_d)
        depthincrease += 1
        node.setPlungeDepth(depthincrease)  # Inside: node.plunge_depth = node.plunge_depth + depthincrease

        resPath = basepath + '/val/problem' + str(index) + '_result.txt'
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
            resPath = basepath + '/val/problem' + str(index) + '_result.txt'
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

        optPath = basepath + '/val/problem' + str(index) + '_optimal.txt'
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
            print("MINLP solver stopped...")
            # 直接对下一个变量进行branch，不剪枝
            ind = node.x_d.size
            node.upper = node.parent.upper
            if ind == num_user0 * num_bs0 or node.upper < globallower:  # 叶子结点或者下界小于上界
                node.fathomed_flag = 1  # 被bb剪枝
                print("Pruning...")
            else:  # 非叶子节点，直接branch
                if node.status == 1:
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
                    resPath = basepath + '/val/problem' + str(index) + '_result.txt'
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

            if node.fathomed_flag != 1:
                tmpPath = basepath + '/tmp.txt'
                tmpFile = open(tmpPath, mode='w')
                feature = np.array(
                    [node.depth / maxdepth, node.upper / root_bound, node.global_lower_bound / root_bound,
                     node.plunge_depth / maxdepth,
                     node.solfound / maxdepth, node.branch, node.sinr])
                # node.power, node.channel])
                # feature = np.array([node.depth/maxdepth,node.upper/root_bound,node.global_lower_bound/root_bound,node.plunge_depth/maxdepth,
                #        node.solfound/maxdepth,node.branch])
                feature_scaled = scaler.transform(feature.reshape(1, -1)).reshape(-1)
                tmpFile.writelines([str(node.status)])
                for j in range(len(feature_scaled)):
                    tmpFile.writelines(['\t', str(j), ':', str(feature_scaled[j])])
                tmpFile.writelines(['\n'])
                tmpFile.close()
                # 用model判断
                modelP, modelF = svm_read_problem(tmpPath)
                p_label, p_acc, p_val = svm_predict(modelP, modelF, model)
                print("Model decision: %d" % p_label[0])

                if node.status == 1:
                    num1 = num1 + 1
                    if p_label[0] == 1:
                        acc1 = acc1 + 1  # branch -> branch
                else:
                    num0 = num0 + 1
                    if p_label[0] == 0:
                        acc0 = acc0 + 1  # prune -> prune

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

    # total_time = time.time() - start_time

    restPath = basepath + '/val/problem_result.txt'
    restFile = open(restPath, mode='r')
    line_temp = restFile.readline()
    line_con = 1
    while line_temp:
        if line_con == index:
            break
        else:
            line_temp = restFile.readline()
            line_con = line_con + 1
    restFile.close()
    temp = (line_temp.strip()).split('\t')
    lower_true = float(temp[2])
    nodenum_true = float(temp[6])
    ogap = (lower_true - globallower) / lower_true
    speed = nodenum_true / nodenum

    return ogap, speed, acc1, acc0, num1, num0
