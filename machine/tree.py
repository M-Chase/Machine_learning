import numpy as np
from sklearn.ensemble import BaggingClassifier
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

plt.style.use('ggplot')
# 画出数据点和边界
def border_of_classifier(sklearn_cl, x, y,index):
        """
        param sklearn_cl : skearn 的分类器
        param x: np.array
        param y: np.array
        """
        ## 1 生成网格数据
        x_min, y_min = x.min(axis = 0)-0.2
        x_max, y_max = x.max(axis = 0)+0.2
        # 利用一组网格数据求出方程的值，然后把边界画出来。
        x_values, y_values = np.meshgrid(np.arange(x_min, x_max, 0.01),
        np.arange(y_min, y_max, 0.01))
        # 计算出分类器对所有数据点的分类结果 生成网格采样
        mesh_output = sklearn_cl.predict(np.c_[x_values.ravel(), y_values.ravel()])
        # 数组维度变形
        mesh_output = mesh_output.reshape(x_values.shape)
        fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
        ## 会根据 mesh_output结果自动从 cmap 中选择颜色
        plt.pcolormesh(x_values, y_values, mesh_output, cmap = 'Accent')
        plt.scatter(x[:, 0], x[:, 1], c = y, s=100, edgecolors ='blue' , linewidth = 1, cmap = plt.cm.Spectral)
        plt.xlim(x_values.min(), x_values.max())
        plt.ylim(y_values.min(), y_values.max())
        # 设置x轴和y轴
        plt.xticks((np.arange(np.ceil(min(x[:, 0])-0.2 ), np.ceil(max(x[:, 0])+0.2 ), 1.0)))
        plt.yticks((np.arange(np.ceil(min(x[:, 1])), np.ceil(max(x[:, 1])), 1.0)))
        # plt.legend('good','bad')
        plt.title('Number='+str(i))
        plt.savefig("tree"+str(index) + ".png")
        plt.show()


filepath = 'F:\课程学习文件\机器学习\实验\watermelon.csv'
dict_data = pd.read_csv(filepath)
print(print(dict_data))
# 转换成np数组
data=np.array(dict_data)
X=data[0:16,0:2]
Y=data[0:16,2:3]
Y=Y.astype(np.int32)
Y=np.squeeze(Y)
list=[3,5,8,12,20]
# map=np.arange()
for i in list:
    clf = BaggingClassifier(n_estimators=i,bootstrap=True)
    clf = clf.fit(X, Y)
    predict=clf.predict(X)
    print(predict, Y)
    border_of_classifier(clf,X,Y,i)





