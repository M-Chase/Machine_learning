import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA
from sklearn.preprocessing import StandardScaler
import os

import skimage.io as io
listdir=r"F:\yalefaces"
filenames=[]
for filename in os.listdir(listdir):
    filenames.append(filename)
imgs=[]
for f in filenames:
    filepath=listdir+"\\"+f
    img = io.imread(filepath)
    imgs.append(img)

imgs=np.array(imgs)
for i in range(1,7):
    plt.subplot(2,3,i), plt.title('image')
    plt.imshow(imgs[i-1:i].squeeze(),cmap='gray'), plt.axis('off')
# plt.savefig("pca"+ ".png")

# plt.show()
imgs=np.reshape(imgs,(165,10000))
estimator = PCA()
estimator.fit(imgs)
new_imgs=estimator.components_
new_imgs=np.reshape(new_imgs,(165,100,100))

plt.figure(figsize=(10,5)) #设置窗口大小
plt.suptitle('Multi_Image') # 图片名称

for j in range(0,5):
    for i in range(1,7):
        plt.subplot(2,3,i), plt.title('image')
        plt.imshow(new_imgs[j*7+i-1:j*7+i].squeeze(),cmap='gray'), plt.axis('off')
    plt.savefig("pca"+ str(j)+".png")
    plt.show()


