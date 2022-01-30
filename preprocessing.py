import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_=None
        self.scale_=None

    def fit(self,X):
        """根据训练数据集X获得数据的均值和方差"""
        assert X.ndim==2,"The dimension of X must be 2"

        self.mean_=np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.mean_=np.array([np.std(X[:,i]) for i in range(X.shape[1])])

        return self

    def transform(self,X):
        """将X根据这个StandardScaler进行均值方差归一化处理"""
        assert X.ndim==2,"The dimension of X must be 2"
        assert self.mean_ is not None and self.scale_ is not None,\
        "must fit before transform!"