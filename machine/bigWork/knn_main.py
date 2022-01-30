import scipy.interpolate as interpolate
from scipy.interpolate import make_interp_spline
from model_selection import train_test_split
from PCA import PCA
from KNN import  KNNClassifier
from metrics import *
from preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from drawPlot import *


tmp = np.loadtxt(r"E:\PyCharm 2019.3.2\WorkSpace\Machine_Work\temp.csv", dtype=np.str, delimiter=",")
datas = tmp[1:,:].astype(np.float)
data=datas[:,:30]
label=datas[:,30:31].astype(np.int)
label=np.squeeze(label)
label[np.where(label==0)]=-1
X_train,Y_train,x_test,y_test=train_test_split(data,label,test_radio=0.2)
stand=StandardScaler()
stand.fit(X_train)
X_train=stand.transform(X_train)
x_test=stand.transform(x_test)

# knn=KNNClassifier(3)
# knn.fit(X_train,Y_train)
# predict=knn.predict(x_test)
#
# score=accuracy_score(y_test,predict)
# mse=mean_squared_error(y_test,predict)
# recall=recall_score(y_test,predict)
# precision=precision_socre(y_test,predict)
# fpr, tpr, thresholds_keras,roc_auc=roc(y_test,predict)
# f1=f1_score(y_test,predict)
# r2=r2_score(y_test,predict)
# tp,fn,fp,tn=confusion_matrix(y_test,predict)
# print("=======================================")
# print("准确率",score)
# print("mse",mse)
# print("召回率",recall)
# print("精确率",precision)
# print("f1score",f1)
# print("R2",r2)
# print("=======================================")
# list=[2,4,6,8,10]
# for i in list:
#     print("PCA降维："+str(i)+"维")
print("=======================================")
pca=PCA(2)
pca.fit(X_train)
x_train_pca=pca.transform(X_train)
x_test_pca=pca.transform(x_test)
knn=KNNClassifier(3)
knn.fit(x_train_pca,Y_train)
predict=knn.predict(x_test_pca)

score=accuracy_score(y_test,predict)
mse=mean_squared_error(y_test,predict)
recall=recall_score(y_test,predict)
precision=precision_socre(y_test,predict)
fpr, tpr, thresholds_keras,roc_auc=roc(y_test,predict)
f1=f1_score(y_test,predict)
r2=r2_score(y_test,predict)
tp,fn,fp,tn=confusion_matrix(y_test,predict)
print("准确率",score)
print("mse",mse)
print("召回率",recall)
print("精确率",precision)
print("f1score",f1)
print("R2",r2)
print("=======================================")

cm=np.array([[tp,fn],[fp,tn]])
plot_confusion_matrix(cm,[1,-1],"Confusion Matrix")
drawROC(fpr, tpr, roc_auc)
drawScatter(x_train_pca,Y_train)
# plotROC(predict,[1,-1])
plt.figure(figsize=(13,8))
plot_decision_region(x_test_pca[0:20],predict[0:20],knn,resolution=0.1)
