#encoding:utf-8

from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.externals import joblib
import datetime
import pandas as pd
import numpy as np
import sys

def in_interval(lower, upper, numb):
    if numb <= upper and numb > lower:
        return True
    else:
        return False

def trainset_split(train_data, split_plot):
    #传入训练集，分割点数组（举例：传入三个分割点那么就有四个子训练集），返回分层后的数据集合
    # 创建长度为子训练集个数的数组
    print ("训练集分割开始")
    sub_trainset_num = len(split_plot)
    print("分割子训练集目标个数:"+str(sub_trainset_num))
    sub_train_data_array = []
    # 将数组的每一个位置初始化为一个数组，每一个数组存储一个子训练集
    for i in range(0, sub_trainset_num):
         sub_train_data_array.append([])

    # 遍历train_data, 根据split_plot中的分割点对数据分层
    for i in range(0, len(train_data)):
        # 最小的一组，第1层
        if train_data[i][0] <= split_plot[0]:
            sub_train_data_array[0].append(train_data[i])
        # 最大的一组，第n层
        elif train_data[i][0] > split_plot[sub_trainset_num-1]:
          sub_train_data_array[sub_trainset_num-1].append(train_data[i])
        else :
            # 对split中的分割点，将区间的上下限和蛋白质长度投到函数中，判断是否属于该区间
            for j in range(0,sub_trainset_num-2):
                if in_interval(split_plot[j],split_plot[j+1], train_data[i][0]) == True:
                    sub_train_data_array[j+1].append(train_data[i])
    # 返回分层好的数据集，格式为[[sub1],[sub2],[sub3],[sub4]...]
    print("分割子训练集实际个数:" + str(len(sub_train_data_array)))
    return sub_train_data_array

def pre_split(pre_data, split_plot):
    #传入训练集，分割点数组，层数
    print("测试集分割开始")
    sub_testset_num = len(split_plot)
    print("测试集分割目标个数:"+str(sub_testset_num))

    # 创建长度为分层数的数组
    sub_pre_data_array = []
    # 创建sequence，记录预测集数据的分组情况
    sequence = []

    # 将数组初始化为层数个数组
    for i in range(0,sub_testset_num):
        sub_pre_data_array.append([])

    # 遍历pre_data,根据split_plot中的分割点对数据分层
    for i in range(0, len(pre_data)):
        # 第一层
        if pre_data[i][0] <= split_plot[0]:
            sub_pre_data_array[0].append(pre_data[i])
            # 记录每条数据对应分到的层
            sequence.append(int(0))
        # 第n层
        elif pre_data[i][0] > split_plot[sub_testset_num-2]:
            sub_pre_data_array[sub_testset_num-1].append(pre_data[i])
            sequence.append(int(sub_testset_num -1))
        else:
            # 遍历split_plot中的分割点，将每条数据分到对应的层中
            for j in range(0,sub_testset_num-1):
                if (in_interval(split_plot[j], split_plot[j+1], pre_data[i][0]) == True):
                    sub_pre_data_array[j+1].append(pre_data[i])
                    sequence.append(int(j+1))
    # 返回分层好的预测集，格式同上；预测数据的分层序列，格式为[1,3,2,4,6...n,n-1,4,2..]
    print("子测试集实际分割数目：" + str(len(sub_pre_data_array)))
    return sub_pre_data_array, sequence


def train_model_for_every_sub_trainset(sub_train_data_array):
    '''
        输入训练集，按层训练回归器并保存模型,并返回模型名称的集合
    '''

    print("模型训练开始，训练模型个数为："+str(len(sub_train_data_array)))

    # 训练并保存好的模型的名称
    train_models_name = []
    for i in range(0,len(sub_train_data_array)):
        # 取其中一层数据
        print("模型" + str(i)+"训练开始")
        train = sub_train_data_array[i]

        #train = pd.DataFrame(train)

        # 划分X，y
        train_X = np.array(train)[:, :-1]
        train_y = np.array(train)[:, -1]

        # 缺失值处理
        imp = Imputer(missing_values="NaN",strategy="mean")
        train_X = imp.fit_transform(train_X)

        #降维
        #pca = PCA(n_components=pca_num)
        #train_X = pca.fit_transform(train_X)

        # 训练
        svr = SVR(kernel='rbf')
        pre_mech = svr.fit(train_X,train_y)

        # 保存模型
        name = 'svr_'+str(i)+'model'
        joblib.dump(pre_mech, name)

        train_models_name.append(name)
        print("模型" + str(i) + "训练结束")

    return train_models_name


def predict_(sub_pre_date_array, train_models_name):
    # 传入分层好的预测集和训练好的模型的名称,加载对应模型进行预测,按分层顺序返回预测值
    print("模型预测开始，预测模型个数为：" + str(len(train_models_name)))

    #预测值
    pre_y_array = []
    for i in range(0,len(sub_pre_date_array)):
        print("模型" + str(i) + "预测开始")
        pre_x = sub_pre_date_array[i]

        # 缺失值处理
        imp = Imputer(missing_values="NaN", strategy="mean")
        pre_x = imp.fit_transform(pre_x)

        # 降维
        #pca = PCA(n_components=pca_num)
        #pre_x = pca.fit_transform(pre_x)

        #加载模型
        name = train_models_name[i]
        pre_model = joblib.load(name)

        #预测一层
        pre_y = pre_model.predict(pre_x)

        pre_y_array.append(pre_y)
        print("模型" + str(i) + "预测结束")

    return pre_y_array

def recovery_result(pre_y_array, affinity_Predict, sequence):
    print("预测集顺序归位开始")
    affinity_Predict_list = affinity_Predict.tolist()
    results = []
    recovery_pre_y = []
    #恢复预测集顺序
    for i in range(0,len(sequence)):
        recovery_pre_y.append(pre_y_array[sequence[i]][0])
        list.remove(pre_y_array[sequence[i]][0])

    #整合输出成文件
    for i in range(0, affinity_Predict_list):
        results.append(affinity_Predict_list[i][0]+affinity_Predict_list[i][1]+recovery_pre_y[i])

    res = pd.DataFrame(results)
    nowTime = datetime.datetime.now().strftime('%m%d%H%M')  # 现在

    filename = 'Result/'+nowTime+'.csv'
    res.to_csv(filename,encoding='utf-8')
    print("预测集顺序归位结束")

if __name__=="__main__":


    train = pd.read_csv('data/trainset_ver2.csv').values
    pre_x = pd.read_csv('data/pre_x_ver2.csv').values

    #数据分层
    sub_train_data_array = trainset_split(train, [200,400,600,800,1000,1300,1500,1800])
    sub_pre_data_array, sequence = pre_split(pre_x,[200,400,600,800,1000,1300,1500,1800])

    train_mech_name = train_model_for_every_sub_trainset(sub_train_data_array)

    pre_y_array = predict_(sub_pre_data_array, train_mech_name)

    Df_affinity_Predict = pd.read_csv('data/df_affinity_test_toBePredicted.csv')
    affinity_Predict = np.array(Df_affinity_Predict)  # np.ndarray()
    recovery_result(pre_y_array, affinity_Predict, sequence)

    print ("预测完毕")
    # train_array = []
    # for i in range(0, 3):
    #     train_array.append([])
    #
    # for i in range(0, 3):
    #     print train_array[i]
