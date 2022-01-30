import numpy as np
import sklearn.datasets as datasets

from PCA import PCA

datas = datasets.load_breast_cancer()
print(datas.keys())
# print(datas.feature_names)
# feature_name=datas.feature_names
# feature_name=list(feature_name)
# feature_name.append("value")
# feature_name=np.array(feature_name)
data=datas.data
pca=PCA(8)
pca.fit(data)
data=pca.transform(data)
label=datas.target.astype(np.int)
label=np.expand_dims(label,axis=1)
all=np.hstack((data,label))
# all=np.vstack((feature_name,all))
np.savetxt("test.csv", all, delimiter=",")