---
title: Chapter01-02 PyTorch的简介和安装、PyTorch基础知识
toc: true
tags: [PyTorch, 组队学习, 笔记]
categories: [04 组队学习, 2021-10 深入浅出PyTorch]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/torch_logo.jpeg?raw=true"
date: 2021-09-18 10:26:03
---
# Chapter01-02 PyTorch的简介和安装、PyTorch基础知识
## 第一章 PyTorch的简介和安装

### PyTorch简介
PyTorch是由Facebook人工智能研究小组开发的一种基于Lua编写的Torch库的Python实现的深度学习库，目前被广泛应用于学术界和工业界，而随着Caffe2项目并入Pytorch， Pytorch开始影响到TensorFlow在深度学习应用框架领域的地位。总的来说，PyTorch是当前难得的简洁优雅且高效快速的框架。因此本课程我们选择了PyTorch来进行开源学习。

### PyTorch的安装
PyTorch官网：[https://pytorch.org/](https://pytorch.org/)

### PyTorch的发展和优势
“All in Pytorch”.

### PyTorch VS TensorFlow

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/03_torch_vs_tf.jpg?raw=true" width="600" alt="" align="center" />

## 第二章 PyTorch的基础知识

### Tensor/张量
张量是基于向量和矩阵的推广，比如我们可以将标量视为零阶张量，矢量可以视为一阶张量，矩阵就是二阶张量。
- 0维张量/标量 标量是1个数字
- 1维张量/向量 1维张量称为“向量”
- 2维张量 2维张量称为“矩阵”
- 3维张量 时间序列数据、股价、文本数据、彩色图片(RGB)
- 4维=图像
- 5维=视频

在PyTorch中， torch.Tensor 是存储和变换数据的主要工具。

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/04_tensor.jpg?raw=true" width="600" alt="" align="center" />

#### tensor-构造

创建一个随机初始化的矩阵：

~~~python
x = torch.rand(4, 3)  # 构造张量
print(x.size())  # 获取维度信息
print(x.shape)  # 获取维度信息
~~~
还有一些常见的构造Tensor的函数：
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/01_tensor_1.jpg?raw=true" width="400" alt="" align="center" />

PyTorch中的Tensor支持超过一百种操作，包括转置、索引、切片、数学运算、线性代数、随机数等等，可参考官方文档。

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/05_tensor.jpg?raw=true" width="600" alt="" align="center" />

#### tensor-squeeze 增加/删除一个维度

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/06_tensor.jpg?raw=true" width="600" alt="" align="center" />

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/07_tensor.jpg?raw=true" width="600" alt="" align="center" />

#### tensor-transpose 转置

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/08_tensor.jpg?raw=true" width="600" alt="" align="center" />

#### tensor-cat concatenate多个tensor

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/09_tensor.jpg?raw=true" width="600" alt="" align="center" />

### 自动求导/自动微分
PyTorch中，所有神经网络的核心是autograd包。autograd包为张量上的所有操作提供了自动求导机制。

#### How to Calculate Gradient

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/10_tensor.jpg?raw=true" width="600" alt="" align="center" />

```python
>>> x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad = True)
>>> x
tensor([[ 1.,  0.],
        [-1.,  1.]], requires_grad=True)
>>> z = x.pow(2)
>>> z
tensor([[1., 0.],
        [1., 1.]], grad_fn=<PowBackward0>)
>>> z = z.sum()
>>> z
tensor(3., grad_fn=<SumBackward0>)
>>> z.backward()
>>> z
tensor(3., grad_fn=<SumBackward0>)
>>> x.grad
tensor([[ 2.,  0.],
        [-2.,  2.]])
```


### 并行计算简介
在利用PyTorch做深度学习的过程中，可能会遇到数据量较大无法在单块GPU上完成，或者需要提升计算速度的场景，这时就需要用到并行计算。
GPU的出现让我们可以训练的更快，更好。PyTorch可以在编写完模型之后，让多个GPU来参与训练。

`CUDA`是我们使用GPU的提供商——NVIDIA提供的GPU并行计算框架。对于GPU本身的编程，使用的是`CUDA`语言来实现的。但是，在我们使用PyTorch编写深度学习代码时，使用的`CUDA`又是另一个意思。在PyTorch使用 `CUDA`表示要开始要求我们的模型或者数据开始使用GPU了。

在编写程序中，当我们使用了 `cuda()` 时，其功能是让我们的模型或者数据迁移到GPU当中，通过GPU开始计算。

不同的数据分布到不同的设备中，执行相同的任务(Data parallelism):
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/pytorch/02_pc_1.png?raw=true" width="250" alt="" align="center" />


## 参考资料
- Datawhale开源项目：深入浅出PyTorch [https://github.com/datawhalechina/thorough-pytorch/](https://github.com/datawhalechina/thorough-pytorch/)
- 李宏毅机器学习2021春-PyTorch Tutorial [https://www.bilibili.com/video/BV1Wv411h7kN?p=5](https://www.bilibili.com/video/BV1Wv411h7kN?p=5)
- What is a gpu and do you need one in deep learning [https://towardsdatascience.com/what-is-a-gpu-and-do-you-need-one-in-deep-learning-718b9597aa0d](https://towardsdatascience.com/what-is-a-gpu-and-do-you-need-one-in-deep-learning-718b9597aa0d)
- 动手学深度学习pytorch版 [https://zh-v2.d2l.ai/chapter_preface/index.html](https://zh-v2.d2l.ai/chapter_preface/index.html)
