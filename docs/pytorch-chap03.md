---
title: Chapter03 PyTorch的主要组成模块
toc: true
tags: [PyTorch, 组队学习, 笔记]
categories: [04 组队学习, 2021-10 深入浅出PyTorch]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/torch_logo.jpeg?raw=true"
date: 2021-09-28 09:31:42
---

# 第三章 PyTorch的主要组成模块

## 完成深度学习的必要部分

机器学习：

1. 数据预处理（数据格式、数据转换、划分数据集）
2. 选择模型，设定损失和优化函数，设置超参数
3. 训练模型，拟合训练集
4. 评估模型，在并在验证集/测试集上计算模型表现

深度学习的注意事项：

1. 数据预处理（数据加载、批处理）
2. 逐层搭建模型，组装不同模块
3. GPU的配置和操作

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/11_dnn.jpg?raw=true" width="600" alt="" align="center" />

## 基本配置

导入必须的包：

```python
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optimizer
```

超参数设置：

```python
batch_size = 16  # batch size
lr = 1e-4  # 初始学习率
max_epochs = 100  # 训练次数 
```

GPU的设置：

```python
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
```

## 数据加载和处理

PyTorch数据读入是通过Dataset+Dataloader的方式完成的，Dataset定义好数据的格式和数据变换形式，Dataloader用iterative的方式不断读入批次数据。

我们可以定义自己的Dataset类来实现灵活的数据读取，定义的类需要继承PyTorch自身的Dataset类。主要包含三个函数：

- `__init__`: 用于向类中传入外部参数，同时定义样本集
- `__getitem__`: 用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
- `__len__`: 用于返回数据集的样本数

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/12_dataset.jpg?raw=true" width="600" alt="" align="center" />

- batch_size：样本是按“批”读入的，batch_size就是每次读入的样本数
- num_workers：有多少个进程用于读取数据
- shuffle：是否将读入的数据打乱
- drop_last：对于样本最后一部分没有达到批次数的样本，不再参与训练

下面是本部分代码在notebook中的运行情况。主要参考 PyTorch官方教程中文版 [https://pytorch123.com/SecondSection/training_a_classifier/](https://pytorch123.com/SecondSection/training_a_classifier/)

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/chap03.jpg?raw=true" width="" alt="" align="center" />

## 模型构建

### 神经网络的构造

PyTorch中神经网络构造一般是基于 Module 类的模型来完成的。Module 类是 nn 模块里提供的一个模型构造类，是所有神经⽹网络模块的基类，我们可以继承它来定义我们想要的模型。下面继承 Module 类构造多层感知机（MLP）。

```python
import torch
from torch import nn

class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Linear(784, 256)
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
    
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)
```

我们可以实例化 MLP 类得到模型变量 net 。下⾯的代码初始化 net 并传入输⼊数据 X 做一次前向计算。其中， net(X) 会调用 MLP 继承⾃自 Module 类的 **call** 函数，这个函数将调⽤用 MLP 类定义的forward 函数来完成前向计算。

```python
>>> import torch
>>> X = torch.rand(2, 784)
>>> X
tensor([[0.3277, 0.2204, 0.5239,  ..., 0.4333, 0.1906, 0.1318],
        [0.9850, 0.2121, 0.8405,  ..., 0.3796, 0.2717, 0.5553]])

>>> from torch import nn
>>> 
>>> class MLP(nn.Module):
...   # 声明带有模型参数的层，这里声明了两个全连接层
...   def __init__(self, **kwargs):
...     # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例例时还可以指定其他函数
...     super(MLP, self).__init__(**kwargs)
...     self.hidden = nn.Linear(784, 256)
...     self.act = nn.ReLU()
...     self.output = nn.Linear(256,10)
...     
...    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
...   def forward(self, x):
...     o = self.act(self.hidden(x))
...     return self.output(o)
... 
>>> net = MLP()
>>> net
MLP(
  (hidden): Linear(in_features=784, out_features=256, bias=True)
  (act): ReLU()
  (output): Linear(in_features=256, out_features=10, bias=True)
)
>>> net(X)
tensor([[ 0.1317,  0.0702,  0.1707, -0.0081, -0.2730,  0.2837,  0.0700,  0.1718,
          0.0299,  0.2082],
        [ 0.1094,  0.0936,  0.2474, -0.0139, -0.1861,  0.1846,  0.1658,  0.2051,
          0.2609,  0.2227]], grad_fn=<AddmmBackward>)
>>> 
```

### 神经网络中常见的层

#### 不含模型参数的层

下⾯构造的 MyLayer 类通过继承 Module 类自定义了一个**将输入减掉均值后输出**的层。这个层里不含模型参数。

```python
>>> import torch
>>> from torch import nn
>>> 
>>> class MyLayer(nn.Module):
...     def __init__(self, **kwargs):
...         super(MyLayer, self).__init__(**kwargs)
...     def forward(self, x):
...         return x - x.mean()  
... 
>>> layer = MyLayer()  # 实例化该层
>>> layer
MyLayer()
>>> layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
tensor([-2., -1.,  0.,  1.,  2.])
```

#### 含模型参数的层

我们还可以自定义含模型参数的自定义层。其中的模型参数可以通过训练学出。

Parameter 类其实是 Tensor 的子类，如果一 个 Tensor 是 Parameter ，那么它会⾃动被添加到模型的参数列表里。所以在⾃定义含模型参数的层时，我们应该将参数定义成 Parameter ，除了直接定义成 Parameter 类外，还可以使⽤ ParameterList 和 ParameterDict 分别定义参数的列表和字典。

```python
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
      
>>> net = MyListDense()
>>> print(net)
MyListDense(
  (params): ParameterList(
      (0): Parameter containing: [torch.FloatTensor of size 4x4]
      (1): Parameter containing: [torch.FloatTensor of size 4x4]
      (2): Parameter containing: [torch.FloatTensor of size 4x4]
      (3): Parameter containing: [torch.FloatTensor of size 4x1]
  )
)
```



```python
class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

>>> net = MyDictDense()
>>> print(net)
MyDictDense(
  (params): ParameterDict(
      (linear1): Parameter containing: [torch.FloatTensor of size 4x4]
      (linear2): Parameter containing: [torch.FloatTensor of size 4x1]
      (linear3): Parameter containing: [torch.FloatTensor of size 4x2]
  )
)
```

下面给出常见的神经网络的一些层，比如卷积层、池化层，以及较为基础的AlexNet，LeNet等。

#### 二维卷积层

二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。

```python
import torch
from torch import nn

# 卷积运算（二维互相关）
def corr2d(X, K): 
    h, w = K.shape
    X, K = X.float(), K.float()
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

填充(padding)是指在输⼊入⾼高和宽的两侧填充元素(通常是0元素)。

在二维互相关运算中，卷积窗口从输入数组的最左上方开始，按从左往右、从上往下 的顺序，依次在输⼊数组上滑动。我们将每次滑动的行数和列数称为步幅(stride)。

（skip）

#### 池化层

池化层每次对输入数据的一个固定形状窗口(⼜称池化窗口)中的元素计算输出。不同于卷积层里计算输⼊和核的互相关性，池化层直接计算池化窗口内元素的最大值或者平均值。该运算也 分别叫做最大池化或平均池化。

```python
>>> import numpy as np
>>> import torch
>>> from torch import nn
>>> 
>>> def pool2d(X, pool_size, mode='max'):
...     p_h, p_w = pool_size
...     Y = np.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
...     for i in range(Y.shape[0]):
...         for j in range(Y.shape[1]):
...             if mode == 'max':
...                 Y[i, j] = X[i: i + p_h, j: j + p_w].max()
...             elif mode == 'avg':
...                 Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
...     return Y
... 
>>> X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
>>> pool2d(X, (2, 2))
array([[4., 5.],
       [7., 8.]])
```

#### 模型示例：LeNet

（待补充）

#### 模型示例：AlexNet

（待补充）

## 损失函数

一个好的训练离不开优质的负反馈，这里的损失函数就是模型的负反馈。

这里将列出PyTorch中常用的损失函数（一般通过torch.nn调用），并详细介绍每个损失函数的功能介绍、数学公式和调用代码。

### 二分类交叉熵损失函数

`torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')`

**功能**：计算二分类任务时的交叉熵（Cross Entropy）函数。在二分类中，label是{0,1}。对于进入交叉熵函数的input为概率分布的形式。一般来说，input为sigmoid激活层的输出，或者softmax的输出。

**主要参数**：

- `weight`:每个类别的loss设置权值
- `size_average`:数据为bool，为True时，返回的loss为平均值；为False时，返回的各样本的loss之和.
- `reduce`:数据类型为bool，为True时，loss的返回是标量。

### 其他损失函数

交叉熵损失函数

L1损失函数

MSE损失函数

平滑L1 (Smooth L1)损失函数

目标泊松分布的负对数似然损失

KL散度

## 优化器

### 什么是优化器

深度学习的目标是通过不断改变网络参数，使得参数能够对输入做各种非线性变换拟合输出，本质上就是一个函数去寻找最优解，只不过这个最优解使一个矩阵。那么我们如何计算出来这么多的系数，有以下两种方法：

1. 第一种是最直接的暴力穷举一遍参数，这种方法的实施可能性基本为0，堪比愚公移山plus的难度。
2. 为了使求解参数过程更加快，人们提出了第二种办法，即就是是BP+优化器逼近求解。

因此，优化器就是根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值，使得模型输出更加接近真实标签。

### PyTorch提供的优化器

Pytorch很人性化的给我们提供了一个优化器的库torch.optim，在这里面给我们提供了十种优化器。

- torch.optim.ASGD
- torch.optim.Adadelta
- torch.optim.Adagrad
- torch.optim.Adam
- torch.optim.AdamW
- torch.optim.Adamax
- torch.optim.LBFGS
- torch.optim.RMSprop
- torch.optim.Rprop
- torch.optim.SGD
- torch.optim.SparseAdam

## 训练与评估

完成了上述设定后就可以加载数据开始训练模型了。首先应该设置模型的状态：如果是训练状态，那么模型的参数应该支持反向传播的修改；如果是验证/测试状态，则不应该修改模型参数。

```python
model.train()   # 训练状态
model.eval()   # 验证/测试状态
```

训练过程：

```python
def train(epoch):
    model.train()
    train_loss = 0
    for data, label in train_loader:  # 此时要用for循环读取DataLoader中的全部数据。
        data, label = data.cuda(), label.cuda()  # 之后将数据放到GPU上用于后续计算，此处以.cuda()为例
        optimizer.zero_grad()  # 开始用当前批次数据做训练时，应当先将优化器的梯度置零
        output = model(data)  # 之后将data送入模型中训练
        loss = criterion(label, output)   # 根据预先定义的criterion计算损失函数
        loss.backward()  # 将loss反向传播回网络
        optimizer.step()  # 使用优化器更新模型参数
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
		print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
```

验证/测试的流程基本与训练过程一致，不同点在于：

- 需要预先设置torch.no_grad，以及将model调至eval模式
- 不需要将优化器的梯度置零
- 不需要将loss反向回传到网络
- 不需要更新optimizer

验证/测试过程：

```python
def val(epoch):       
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
            running_accu += torch.sum(preds == label.data)
    val_loss = val_loss/len(val_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, val_loss))
```

## 可视化

在PyTorch深度学习中，可视化是一个可选项，指的是某些任务在训练完成后，需要对一些必要的内容进行可视化，比如分类的ROC曲线，卷积网络中的卷积核，以及训练/验证过程的损失函数曲线等等。

## 参考资料
- Datawhale开源项目：深入浅出PyTorch [https://github.com/datawhalechina/thorough-pytorch/](https://github.com/datawhalechina/thorough-pytorch/)
- 李宏毅机器学习2021春-PyTorch Tutorial [https://www.bilibili.com/video/BV1Wv411h7kN?p=5](https://www.bilibili.com/video/BV1Wv411h7kN?p=5)
- 动手学深度学习pytorch版 [https://zh-v2.d2l.ai/chapter_preface/index.html](https://zh-v2.d2l.ai/chapter_preface/index.html)
- PyTorch官方教程中文版 [https://pytorch123.com/SecondSection/training_a_classifier/](https://pytorch123.com/SecondSection/training_a_classifier/)

