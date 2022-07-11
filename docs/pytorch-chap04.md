---
title: Chapter04 PyTorch基础实战——FashionMNIST图像分类
toc: true
tags: [PyTorch, team learning, note]
categories: [04 组队学习, 2021-10 深入浅出PyTorch]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/torch_logo.jpeg?raw=true"
date: 2021-10-14 18:02:55
---
# 第四章 PyTorch基础实战——FashionMNIST图像分类

## 数据集和任务介绍

我们这里的任务是对10个类别的“时装”图像进行分类，使用FashionMNIST数据集。

FashionMNIST数据集中包含已经预先划分好的训练集和测试集，其中训练集共60,000张图像，测试集共10,000张图像。每张图像均为单通道黑白图像，大小为32*32pixel，分属10个类别。
## 导入必要的包

```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
```

## 配置训练环境和超参数

```python
# 配置GPU，这里有两种方式
## 方案一：使用os.environ
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

## 配置其他超参数，如batch_size, num_workers, learning rate, 以及总的epochs
batch_size = 256
num_workers = 4
lr = 1e-4
epochs = 20
```

## 数据读入和加载

数据读入有两种方式:

- 下载并使用PyTorch提供的内置数据集。这种方式只适用于常见的数据集，如MNIST，CIFAR10等，PyTorch官方提供了数据下载。这种方式往往适用于快速测试方法（比如测试下某个idea在MNIST数据集上是否有效）
- 从网站下载以csv格式存储的数据，读入并转成预期的格式。这种数据读入方式需要自己构建Dataset，这对于PyTorch应用于自己的工作中十分重要

同时，还需要对数据进行必要的变换，比如说需要将图片统一为一致的大小，以便后续能够输入网络训练；需要将数据格式转为Tensor类，等等。这些变换可以很方便地借助torchvision包来完成，torchvision这是PyTorch官方用于图像处理的工具库。

```python
# 首先设置数据变换
from torchvision import transforms

image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),   # 这一步取决于后续的数据读取方式，如果使用内置数据集则不需要
    transforms.Resize(image_size),
    transforms.ToTensor()
])
```

读取方式一：

```python
## 读取方式一：使用torchvision自带数据集，下载可能需要一段时间
from torchvision import datasets

train_data = datasets.FashionMNIST(root='./', train=True, download=True, transform=data_transform)
test_data = datasets.FashionMNIST(root='./', train=False, download=True, transform=data_transform)
```

读取方式二：

```python
## 读取方式二：读入csv格式的数据，自行构建Dataset类
class FMDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28,28,1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

train_df = pd.read_csv("./FashionMNIST/fashion-mnist_train.csv")
test_df = pd.read_csv("./FashionMNIST/fashion-mnist_test.csv")
train_data = FMDataset(train_df, data_transform)
test_data = FMDataset(test_df, data_transform)
```

> 注意：这里需要自己下载数据。可以从kaggle上下载（需科学上网）（但貌似也不是教程用的版本）：[https://www.kaggle.com/zalando-research/fashionmnist/](https://www.kaggle.com/zalando-research/fashionmnist/)

```python
# 定义DataLoader类，以便在训练和测试时加载数据
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
```

```python
# 数据可视化
import matplotlib.pyplot as plt
image, label = next(iter(test_loader))
print(image.shape, label.shape)
plt.imshow(image[0][0], cmap="gray")
```

> 这里程序运行了很久，一直跑不出结果，改用了colab

## 模型设计

手搭一个CNN

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x

model = Net()
# model = model.cuda()  # 将模型放到GPU上用于训练
# model = nn.DataParallel(model).cuda()   # 多卡训练时的写法，之后的课程中会进一步讲解
```

## 设定损失函数

使用torch.nn模块自带的CrossEntropy损失。
PyTorch会自动把整数型的label转为one-hot型，用于计算CE loss。
这里需要确保label是从0开始的，同时模型不加softmax层（使用logits计算）,这也说明了PyTorch训练中各个部分不是独立的，需要通盘考虑。

```python
criterion = nn.CrossEntropyLoss()
# criterion = nn.CrossEntropyLoss(weight=[1,1,1,1,3,1,1,1,1,1])
```

## 设定优化器

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## 训练和测试

**训练和测试（验证）**
各自封装成函数，方便后续调用
关注两者的主要区别：

- 模型状态设置
- 是否需要初始化优化器
- 是否需要将loss传回到网络
- 是否需要每步更新optimizer

此外，对于测试或验证过程，可以计算分类准确率

```python
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:
        # data, label = data.cuda(), label.cuda()  # 不用cuda先
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
```

```python
def val(epoch):       
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            # data, label = data.cuda(), label.cuda()  # 不用cuda先
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
```

```python
for epoch in range(1, epochs+1):
    train(epoch)
    val(epoch)
```

结果（不是很好）：

```python
/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
/usr/local/lib/python3.7/dist-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
Epoch: 1 	Training Loss: 1.859782
Epoch: 1 	Validation Loss: 1.252422, Accuracy: 0.504242
Epoch: 2 	Training Loss: 1.073511
Epoch: 2 	Validation Loss: 0.958262, Accuracy: 0.620891
Epoch: 3 	Training Loss: 0.912065
Epoch: 3 	Validation Loss: 0.859967, Accuracy: 0.682927
Epoch: 4 	Training Loss: 0.803673
Epoch: 4 	Validation Loss: 0.725328, Accuracy: 0.743902
Epoch: 5 	Training Loss: 0.723244
Epoch: 5 	Validation Loss: 0.699738, Accuracy: 0.725345
Epoch: 6 	Training Loss: 0.676728
Epoch: 6 	Validation Loss: 0.688325, Accuracy: 0.742312
Epoch: 7 	Training Loss: 0.624213
Epoch: 7 	Validation Loss: 0.633743, Accuracy: 0.744963
Epoch: 8 	Training Loss: 0.595873
Epoch: 8 	Validation Loss: 0.588029, Accuracy: 0.770414
Epoch: 9 	Training Loss: 0.561574
Epoch: 9 	Validation Loss: 0.578903, Accuracy: 0.765642
Epoch: 10 	Training Loss: 0.544152
Epoch: 10 	Validation Loss: 0.563249, Accuracy: 0.791622
Epoch: 11 	Training Loss: 0.532662
Epoch: 11 	Validation Loss: 0.561163, Accuracy: 0.790032
Epoch: 12 	Training Loss: 0.520769
Epoch: 12 	Validation Loss: 0.560051, Accuracy: 0.783139
Epoch: 13 	Training Loss: 0.495388
Epoch: 13 	Validation Loss: 0.537520, Accuracy: 0.794804
Epoch: 14 	Training Loss: 0.461928
Epoch: 14 	Validation Loss: 0.533855, Accuracy: 0.799046
Epoch: 15 	Training Loss: 0.453786
Epoch: 15 	Validation Loss: 0.534338, Accuracy: 0.805408
Epoch: 16 	Training Loss: 0.457692
Epoch: 16 	Validation Loss: 0.515626, Accuracy: 0.812831
Epoch: 17 	Training Loss: 0.449596
Epoch: 17 	Validation Loss: 0.504590, Accuracy: 0.816013
Epoch: 18 	Training Loss: 0.443980
Epoch: 18 	Validation Loss: 0.503526, Accuracy: 0.818134
Epoch: 19 	Training Loss: 0.420621
Epoch: 19 	Validation Loss: 0.488520, Accuracy: 0.826087
Epoch: 20 	Training Loss: 0.418917
Epoch: 20 	Validation Loss: 0.524965, Accuracy: 0.797985
```





## 参考资料

- 程序的colab链接：[https://colab.research.google.com/drive/1kvaBEEgQ_a5G5xOHUe5ih4Qq8L5C9zYY?usp=sharing](https://colab.research.google.com/drive/1kvaBEEgQ_a5G5xOHUe5ih4Qq8L5C9zYY?usp=sharing)
- Datawhale开源项目：深入浅出PyTorch [https://github.com/datawhalechina/thorough-pytorch/](https://github.com/datawhalechina/thorough-pytorch/)
- 李宏毅机器学习2021春-PyTorch Tutorial [https://www.bilibili.com/video/BV1Wv411h7kN?p=5](https://www.bilibili.com/video/BV1Wv411h7kN?p=5)
- 动手学深度学习pytorch版 [https://zh-v2.d2l.ai/chapter_preface/index.html](https://zh-v2.d2l.ai/chapter_preface/index.html)
- PyTorch官方教程中文版 [https://pytorch123.com/SecondSection/training_a_classifier/](https://pytorch123.com/SecondSection/training_a_classifier/)
