import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # 背景色
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # 这里+1  -1的操作我理解为防止样本落在图的边缘处
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # print(x1_min, x1_max)

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # print(x2_min, x2_max)

    # 生成网格点坐标矩阵
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    # 绘制轮廓等高线  alpha参数为透明度
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')
    plt.show()

def drawROC(fpr, tpr,roc_auc):
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()

def drawScatter(x_train_pca,Y_train,flag=False,support_xlim=None,support_ylim=None,x=None,y=None):
    plt.scatter(x_train_pca[np.where(Y_train == 1)][:, 0], x_train_pca[np.where(Y_train == 1)][:, 1], color='b')
    plt.scatter(x_train_pca[np.where(Y_train == -1)][:, 0], x_train_pca[np.where(Y_train == -1)][:, 1], color='r')
    if flag:
        x.shape = (len(x), 1)
        y.shape = (len(x), 1)
        plt.plot(x, y, color="darkorange", linewidth=3.0, label="Boarder")  # 继续在ax图上作图
    plt.show()

def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='r')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b-')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    print("the Area Under the Curve is: ", ySum * xStep)
    plt.show()