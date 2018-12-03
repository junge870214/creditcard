#主要模块导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #可视化
import seaborn as sns #封装matplotlib的可视化包
from sklearn.preprocessing import StandardScaler #标准化函数
from sklearn.model_selection import KFold#交叉验证
from sklearn.linear_model import LogisticRegression#导入逻辑回归模型
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix#导入指标，查准率，查全率，f1,混淆矩阵
#1数据导入
data = pd.read_csv('/Users/wujie/学习/机器学习/python/01、【非加密】python数据分析与机器学习实战/课程资料/唐宇迪-机器学习课程资料/机器学习算法配套案例实战/逻辑回归-信用卡欺诈检测/creditcard.csv')
#观察数据，是否存在数据不对称，特别是分类问题。考虑采用，欠采样，过采样的方法
data.head()
data.shape
num = pd.value_counts(y['Class'],sort='True').sort_index()
#绘制盒图，观察每个特征的取值情况
fig = plt.figure(figsize = (10,5))
sns.boxplot(data=data)
#做归一化处理
ss = StandardScaler()
data['normAmount'] = ss.fit_transform(np.array(data['Amount']).reshape(-1,1))
data = data.drop(['Time','Amount'],axis =1)
#下采样法
num_of_1 = len(data[data.Class == 1])#获取class = 1 的个数
index_of_1 = np.array(data[data.Class == 1].index) #获取class = 1 的索引
#从class = 0的样本中选出num_of_1个，组成样本
index_of_0_before = data[data.Class == 0].index
index_of_0 = np.array(np.randam.choice(index_of_0_before,num_of_1,replace = Flase))
#组成样本数据
index_of_sample = np.concatenate([index_of_1,index_of_0])
sample = data.iloc[index_of_sample,:].sort_index()
#交叉验证采样
'''
概述
scikit-learn中，与逻辑回归有关的主要是这三个类：LogisticRegression,LogisticRegressionCV和logistic_regression_path。
其中LogisticRegression和LogisticRegressionCV的主要区别是CV使用了交叉验证来选择正则化系数C，其他使用区别不大。
正则化选择参数：penalty
penalty可选值为"l1","l2"，默认l2。
只为了解决过拟合，直接选l2就够了，如果选了l2后还是过拟合，可以考虑l1，模型特征较多，希望让模型系数稀疏化，可以考虑l1
penalty的选取会影响损失函数优化算法的选取，即：solver参数。
如果选l2正则化，四种优化算法：newton-cg，lbfgs，liblinear，sag都可选。
选l1正则化，只能选liblinear，因为其与三种优化算法都需要损失函数的一阶二阶连续可导。
solver参数：
liblinear：使用开源liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数
lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
newton-cg：牛顿法家族的一种，利用损失函数二阶导数即海森矩阵来优化迭代损失函数
sag：随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适用于样本数据较多的情况
分类参数：muiti_class
multi_class参数决定了我们分类方式的选取，默认为ovr
在二元问题上没有区别，区别在多元逻辑回归上。
类型权重参数：class_weight
class_weight表示分类模型各种类型的权重，主要解决：误分类代价很高，样本高度失衡
calss_weight可选balanced，让库自己计算权重，样本量越多，权重越低。
样本权重参数：sample_weight
'''

fold = KFold(5,shuffle = True)#shuffle为True代表顺序打乱，建立K折交叉验证
for train_index,test_index in fold.split(sample):#获取交叉验证索引，每次不一样，轮流n次
    train = sample.iloc[train_index,:]
    test=sample.iloc[test_index,:]
    train_x=train.iloc[:,train.columns!='Class']
    train_y=train.iloc[:,train.columns=='Class']
    test_x=test.iloc[:,train.columns!='Class']
    test_y=test.iloc[:,train.columns=='Class']
    model.fit(train_x.values,train_y.values.ravel())
    predict_y=model.predict_proba(test_x)#获取概率值
    predict_rate,recall_rate,thresholds = predict_recall_curve(test_y,predict_y[:,1],pos_label=1)#绘制P-R曲线
    false_positive_rate,true_positive_rate,thresholds = roc_curve(test_y,predict_y[:,1],pos_label=1)#绘制ROC曲线
 
#绘制混淆矩阵函数
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')