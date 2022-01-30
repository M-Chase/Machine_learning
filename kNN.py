import numpy as np
from math import sqrt
from collections import Counter

def kNN_classify(k,X_train,y_train,x):
    # 进行断言，确定合理性
    assert 1<= k<=X_train.shape[0],"k must be valid"
    assert X_train.shape[0]==y_train.shape[0],\
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1]==x.shape[0],\
        "the feature number of x must be equal to X_train"
    #求出所有点离样品的距离
    distances=[sqrt(np.sum((x_train-x)**2)) for x_train in X_train]
    # 找出最近的几个点
    nearset=np.argsort(distances)

    # 求出那几个点所对应的y值
    topK_y=[y_train[i] for i in nearset[:k]]
    # 用Counter进行计数
    votes=Counter(topK_y)
    # 返回计数最大值作为结果
    return votes.most_common(1)[0][0]