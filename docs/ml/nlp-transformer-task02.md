---
title: Task02 学习Attentioin和Transformer
toc: true
tags: [NLP, 预训练模型, 组队学习, attention, transfomer, 笔记]
categories: [04 组队学习, 2021-09 基于transformer的NLP]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/nlp/00_bg.png?raw=true"
date: 2021-09-17 09:09:24
---
# Task02 学习Attentioin和Transformer
## Attention
### seq2seq
seq2seq是一种常见的NLP模型结构，全称是：sequence to sequence，翻译为“序列到序列”。顾名思义：从一个文本序列得到一个新的文本序列。典型的任务有：机器翻译任务，文本摘要任务。

seq2seq模型由编码器（encoder）和解码器（decoder）组成，编码器用来分析输入序列，解码器用来生成输出序列。编码器会处理输入序列中的每个元素，把这些信息转换成为一个背景向量（context vector）。当我们处理完整个输入序列后，编码器把背景向量发送给解码器，解码器通过背景向量中的信息，逐个元素输出新的序列。

**在transformer模型之前，seq2seq中的编码器和解码器一般采用循环神经网络（RNN）**，虽然非常经典，但是局限性也非常大。最大的局限性就在于编码器和解码器之间的唯一联系就是一个固定长度的context向量。也就是说，编码器要将整个序列的信息压缩进一个固定长度的向量中。这样做存在两个弊端：
- 语义向量可能无法完全表示整个序列的信息
- 先输入到网络的内容携带的信息会被后输入的信息覆盖掉，输入序列越长，这个现象就越严重

### Attention
为了解决seq2seq模型中的两个弊端，Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》中提出使用Attention机制，使得seq2seq模型可以有区分度、有重点地关注输入序列，从而极大地提高了机器翻译的质量。

一个有注意力机制的seq2seq与经典的seq2seq主要有2点不同：
1. 首先，编码器会把更多的数据传递给解码器。编码器把所有时间步的 hidden state（隐藏层状态）传递给解码器，而不是只传递最后一个 hidden state（隐藏层状态）
2. 注意力模型的解码器在产生输出之前，做了一个额外的attention处理

## Transformer
### 模型架构
transformer原论文的架构图：

<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/trm_1.png?raw=true" width="400" alt="" align="center" />

一个更清晰的架构图：
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/trm_2.png?raw=true" width="600" alt="" align="center" />

从输入到输出拆开看就是：
- INPUT：input vector + position encoding
- ENCODERs（×6），and each encoder includes：
  - input
  - multi-head self-attention
  - residual connection&norm
  - full-connected network
  - residual connection&norm
  - output
- DECODERs（×6），and each decoder includes：
  - input 
  - Masked multihead self-attention
  - residual connection&norm
  - multi-head self-attention
  - residual connection&norm
  - full-connected network
  - residual connection&norm
  - output
- OUTPUT：
  - output (decoder's)
  - linear layer
  - softmax layer
  - output


### 模型输入
#### 词向量
和常见的NLP任务一样，我们首先会使用词嵌入算法（embedding），将输入文本序列的每个词转换为一个词向量。

#### 位置向量
Transformer模型对每个输入的词向量都加上了一个位置向量。这些向量有助于确定每个单词的位置特征，或者句子中不同单词之间的距离特征。词向量加上位置向量背后的直觉是：将这些表示位置的向量添加到词向量中，得到的新向量，可以为模型提供更多有意义的信息，比如词的位置，词之间的距离等。

*（生成位置编码向量的方法有很多种）*

### 编码器和解码器
*注：1. 编码器和解码器中有相似的模块和结构，所以合并到一起介绍。*
*2. 本部分按照李宏毅老师的Attention，Transformer部分的课程PPT来，因为lee的课程对新手更友好。*

#### Self-Attention
self-attention对于每个向量都会考虑整个sequence的信息后输出一个向量，self-attention结构如下：
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/04_attention_1.png?raw=true" width="600" alt="" align="center" />
FC：Fully-connected network 全连接网络
ai: 输入变量。可能是整个网络的输入，也可能是某个隐藏层的输出
bi: 考虑整个sequence信息后的输出变量

矩阵计算：
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/13_matrix_4.jpg?raw=true" width="300" alt="" align="center" />
目标：根据输入向量矩阵I，计算输出向量矩阵O。矩阵运算过程：
1. 矩阵I分别乘以Wq, Wk, Wv（参数矩阵，需要模型进行学习），得到矩阵Q, K, V。
2. 矩阵K的转置乘以Q，得到注意力权重矩阵A，归一化得到矩阵A’。
3. 矩阵V乘矩阵A‘，得到输出向量矩阵O。

#### Multi Head Self-Attention
*简单地说，多了几组Q，K，V。在Self-Attention中，我们是使用𝑞去寻找与之相关的𝑘，但是这个相关性并不一定有一种。那多种相关性体现到计算方式上就是有多个矩阵𝑞，不同的𝑞负责代表不同的相关性。*

Transformer 的论文通过增加多头注意力机制（一组注意力称为一个 attention head），进一步完善了Self-Attention。这种机制从如下两个方面增强了attention层的能力：
- 它扩展了模型关注不同位置的能力。
- 多头注意力机制赋予attention层多个“子表示空间”。

#### 残差链接和归一化
残差链接：一种把input向量和output向量直接加起来的架构。
归一化：把数据映射到0～1范围之内处理。

### 模型输出
#### 线性层和softmax
Decoder 最终的输出是一个向量，其中每个元素是浮点数。我们怎么把这个向量转换为单词呢？这是线性层和softmax完成的。

线性层就是一个普通的全连接神经网络，可以把解码器输出的向量，映射到一个更大的向量，这个向量称为 logits 向量：假设我们的模型有 10000 个英语单词（模型的输出词汇表），此 logits 向量便会有 10000 个数字，每个数表示一个单词的分数。

然后，Softmax 层会把这些分数转换为概率（把所有的分数转换为正数，并且加起来等于 1）。然后选择最高概率的那个数字对应的词，就是这个时间步的输出单词。

#### 损失函数
Transformer训练的时候，需要将解码器的输出和label一同送入损失函数，以获得loss，最终模型根据loss进行方向传播。

只要Transformer解码器预测了组概率，我们就可以把这组概率和正确的输出概率做对比，然后使用反向传播来调整模型的权重，使得输出的概率分布更加接近整数输出。

那我们要怎么比较两个概率分布呢？：我们可以简单的用两组概率向量的的空间距离作为loss（向量相减，然后求平方和，再开方），当然也可以使用交叉熵(cross-entropy)]和KL 散度(Kullback–Leibler divergence)。

## 参考资料

**理论部分**
[1] (强推)李宏毅2021春机器学习课程 [https://www.bilibili.com/video/BV1Wv411h7kN?from=search&seid=17090062977285779802&spm_id_from=333.337.0.0](https://www.bilibili.com/video/BV1Wv411h7kN?from=search&seid=17090062977285779802&spm_id_from=333.337.0.0)
[2] **基于transformers的自然语言处理(NLP)入门（涵盖了图解系列、annotated transformer、huggingface）** [https://github.com/datawhalechina/learn-nlp-with-transformers](https://github.com/datawhalechina/learn-nlp-with-transformers)
[3] 图解transformer|The Illustrated Transformer [http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)
[4] 图解seq2seq, attention|Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention) [https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

**代码部分**
[5] The Annotated Transformer [http://nlp.seas.harvard.edu//2018/04/03/attention.html](http://nlp.seas.harvard.edu//2018/04/03/attention.html)
[6] Huggingface/transformers [https://github.com/huggingface/transformers/blob/master/README_zh-hans.md](https://github.com/huggingface/transformers/blob/master/README_zh-hans.md)

**论文部分**
Attention is all "we" need.

**其他不错的博客或教程**
[7] 基于transformers的自然语言处理(NLP)入门--在线阅读 [https://datawhalechina.github.io/learn-nlp-with-transformers/#/](https://datawhalechina.github.io/learn-nlp-with-transformers/#/)
[8] 李宏毅2021春机器学习课程笔记——自注意力机制 [https://www.cnblogs.com/sykline/p/14730088.html](https://www.cnblogs.com/sykline/p/14730088.html)
[9] 李宏毅2021春机器学习课程笔记——Transformer模型 [https://www.cnblogs.com/sykline/p/14785552.html](https://www.cnblogs.com/sykline/p/14785552.html)
[10] 李宏毅机器学习学习笔记——自注意力机制 [https://blog.csdn.net/p_memory/article/details/116271274](https://blog.csdn.net/p_memory/article/details/116271274)
[11] 车万翔-自然语言处理新范式：基于预训练的方法【讲座+PPT】 [https://app6ca5octe2206.pc.xiaoe-tech.com/detail/v_611f48f3e4b02ac39d12246f/3?fromH5=true](https://app6ca5octe2206.pc.xiaoe-tech.com/detail/v_611f48f3e4b02ac39d12246f/3?fromH5=true)
[12] 苏剑林-《Attention is All You Need》浅读（简介+代码）[https://spaces.ac.cn/archives/4765](https://spaces.ac.cn/archives/4765)
