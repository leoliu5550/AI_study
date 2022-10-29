from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
"""
# MetaData
{
    b'num_cases_per_batch': 10000, 
    b'label_names': 
    [
        b'airplane', 
        b'automobile', 
        b'bird', 
        b'cat', 
        b'deer', 
        b'dog', 
        b'frog', 
        b'horse', 
        b'ship', 
        b'truck'
        ], 
    b'num_vis': 3072
}
"""

class CDataset(Dataset):
    def __init__(self,data_dir):
        '''
        大概有兩種用法
        1.把所有要用的資料一次性的都存進來
        2.當資料太大，會塞爆記憶體的時候，可以定義文件的索引就好，在真的要用資料的時候才根據索引讀取資料
            2.比起1. 較慢
        '''
        self.data_dir = data_dir
        self.meda_data = 'batches.meta'
        self.bath_train_list = ['data_batch_'+str(i) for i in range(1,6,1)]
        self.bath_test = 'test_batch'

    def __getitem__(self,idx):
        bat = (idx//10000)
        idx = (idx % 10000)
        path = os.path.join(
            self.data_dir,
            self.bath_train_list[bat]
        )
        
        data = self.unpickle(path)
        img = np.transpose(np.reshape(data[b'data'][idx],(3, 32,32)), (1,2,0))
        print(img.shape)
        # data prepare
        img_tensor = torch.from_numpy(img)
        print(img_tensor.shape)
        label = data[b'labels'][idx]
        label_name = data[b'filenames'][idx]
        
        return label,label_name,img_tensor
    
    def __len__(self):
    # 需要返回資料的長度，取資料的時候才知道要找哪個
        return 10000*5
    
    # 一些可能會用到的工具
    def metadata(self):
        path = os.path.join(self.data_dir,self.meda_data)
        return self.unpickle(path)

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    @staticmethod
    def show_image(tensor,name):
        plt.title(name)
        img = np.transpose(np.reshape(tensor,(3, 32,32)), (1,2,0))
        plt.imshow(img)
        plt.show()

        return img

def timer(func):
    def warp(*args):
        time_start = time.time()
        func(*args)
        time_end = time.time() 
        time_c= time_end - time_start
        print(f"{func.__name__} cost is {time_c}")
        return 

    return warp
datasets = CDataset(r"DATA/cifar-10-batches")
# data = data_geter.unpickle("DATA/cifar-10-batches/data_batch_5")
# print(data[b'data'][0])
# print(data.keys())
# data_geter.show_image(data[b'data'][10],data[b'filenames'][10])
dataloader = DataLoader(
    dataset=datasets, 
    batch_size=1, 
    shuffle=False)

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
            )
        self.sig = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=2
            )
        self.cnn2 = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.sig2 = nn.Sigmoid()
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2
            ) 
        self.fc = nn.Linear(7*7*1,10)
        self.fc2 = nn.Linear(10,10)
    
    def forward(self, x):
        
        out = self.cnn1(x)
        out = self.sig(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.sig2(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.fc2(out)
        return out
# Traning the Model
#history-like list for store loss & acc value
training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

CNN = cnn()
print(CNN)



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(CNN.parameters(), lr=0.1)

EPOCH = 1

for epoch in range(EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        labels,names ,inputs  = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = CNN(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0
        break

print('Finished Training')

PATH = './model/cifar_net.pth'
torch.save(CNN.state_dict(), PATH)