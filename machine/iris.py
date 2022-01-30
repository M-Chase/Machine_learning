import torch
from torch import nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,TensorDataset

EPOCH =500               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.003              # learning rate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iris=load_iris()
print(iris['target_names'])
print(iris['data'])
print(iris['DESCR'])
train_data,test_data,train_label,test_label=train_test_split(iris.data,iris.target,train_size=0.7)
# train_data=iris.data[0:150,:]
# train_label=iris.target[0:150]
# test_data=iris.data[0:,:]
# test_label=iris.target[0:]


X_train=torch.FloatTensor(train_data).to(device)
Y_train=torch.LongTensor(train_label).to(device)
X_test=torch.FloatTensor(test_data).to(device)
Y_test=torch.LongTensor(test_label).to(device)

data_train=TensorDataset(X_train,Y_train)
data_test=TensorDataset(X_test,Y_test)

train_loader=DataLoader(dataset=data_train,batch_size=BATCH_SIZE, shuffle=True)
test_loader=DataLoader(dataset=data_test,batch_size=BATCH_SIZE, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc=nn.Linear(4,3)
        self.out=nn.Softmax(dim=1)
    def forward(self,x):
        x = x.view(x.size(0), -1)
        x=self.fc(x)
        out=self.out(x)
        return out
net=Net().to(device)
optimizer=torch.optim.Adam(net.parameters(),lr=0.003)
loss_func=nn.CrossEntropyLoss()

def train(epoch):
    for step,(x,y)in enumerate(train_loader):
        output=net(x)
        loss=loss_func(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(len(x))
        if step%3==0:
            print('Train Epoch:{}[{}/{}({:.0f})]\tLoss:{:.6f}'.format(
                epoch,step*len(x),len(train_loader.dataset),
                100.*step/len(train_loader),loss.item()))

def test():
    test_loss=0
    correct=0
    for x,y in test_loader:
        output=net(x)
        test_loss+=loss_func(output,y).item()
        pred=torch.max(output.data,1)[1]
        correct+=pred.eq(y.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1,20):
    train(epoch)
    test()

# test_output=net(X_train)
# pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
# print(pred_y)
# print(Y_train)