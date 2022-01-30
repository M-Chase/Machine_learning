import numpy as np
from math import sqrt
from sklearn.metrics import roc_curve, auc

def accuracy_score(y_true,y_predict):
    """计算y_true和y_predict之间的准确度"""
    assert y_true.shape[0]==y_predict.shape[0],\
    "the size of y_true must be equal to the size of y_predict"

    return sum(y_true==y_predict)/len(y_true)

def precision_socre(y_true,y_predict):
    """正确被检索的item(TP)"占所有"实际被检索到的(TP+FP)"的比例"""
    y_predict_1 = np.where(y_predict == 1)[0]
    y_true_1 = np.where(y_true == 1)[0]

    tp = len(np.where(np.in1d(y_predict_1, y_true_1))[0])
    tp_fp=sum(y_predict[np.where(y_predict==1)])

    return tp/tp_fp

def f1_score(y_true,y_predict):
    y_predict_1=np.where(y_predict == 1)[0]
    y_predict_0=np.where(y_predict == -1)[0]
    y_true_1 = np.where(y_true == 1)[0]
    y_true_0 = np.where(y_true == -1)[0]

    tp=len(np.where(np.in1d(y_predict_1, y_true_1))[0])
    tn=len(np.where(np.in1d(y_predict_0, y_true_0))[0])
    return 2*tp/(len(y_predict)+tp-tn)

def recall_score(y_true,y_predict):
    y_predict_1 = np.where(y_predict == 1)[0]
    y_true_1 = np.where(y_true == 1)[0]
    tp=len(np.where(np.in1d(y_predict_1, y_true_1))[0])
    tp_fn = sum(y_true[np.where(y_true == 1)])

    return tp / tp_fn

def roc(y_true,y_predict):
    fpr, tpr, thresholds_keras=roc_curve(y_true, y_predict,pos_label=1)
    roc_auc=auc(fpr,tpr)
    return fpr, tpr, thresholds_keras,roc_auc

def confusion_matrix(y_true,y_predict):
    y_predict_1=np.where(y_predict == 1)[0]
    y_predict_0=np.where(y_predict == -1)[0]
    y_true_1 = np.where(y_true == 1)[0]
    y_true_0 = np.where(y_true == -1)[0]
    tp=len(np.where(np.in1d(y_predict_1, y_true_1))[0])
    tn=len(np.where(np.in1d(y_predict_0, y_true_0))[0])
    fn=len(np.where(np.in1d(y_predict_0, y_true_1))[0])
    fp = len(np.where(np.in1d(y_predict_1, y_true_0))[0])
    return tp,fn,fp,tn
def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的MSE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_true - y_predict)**2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""

    return sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的RMSE"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def r2_score(y_true, y_predict):
    """计算y_true和y_predict之间的R Square"""

    return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)