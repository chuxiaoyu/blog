---
title: Task03 学习BERT
toc: true
tags: [NLP, 预训练模型, 组队学习, BERT, 笔记]
categories: [04 组队学习, 2021-09 基于transformer的NLP]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/nlp/00_bg.png?raw=true"
date: 2021-09-17 09:44:18
---
# Task03 学习BERT
## BERT简介
BERT首先在大规模无监督语料上进行预训练，然后在预训练好的参数基础上增加一个与任务相关的神经网络层，并在该任务的数据上进行微调训，最终取得很好的效果。**BERT的这个训练过程可以简述为：预训练（pre-train）+微调（fine-tune/fine-tuning），已经成为最近几年最流行的NLP解决方案的范式。**

### 如何直接应用BERT
1. 下载在无监督语料上预训练好的BERT模型，一般来说对应了3个文件：BERT模型配置文件（用来确定Transformer的层数，隐藏层大小等），BERT模型参数，BERT词表（BERT所能处理的所有token）。
2. 针对特定任务需要，在BERT模型上增加一个任务相关的神经网络，比如一个简单的分类器，然后在特定任务监督数据上进行微调训练。（微调的一种理解：学习率较小，训练epoch数量较少，对模型整体参数进行轻微调整）

### BERT的结构
**BERT模型结构基本上就是Transformer的encoder部分。**

### BERT的输入和输出
BERT模型输入有一点特殊的地方是在一句话最开始拼接了一个[CLS] token，如下图所示。这个特殊的[CLS] token经过BERT得到的向量表示通常被用作当前的句子表示。我们直接使用第1个位置的向量输出（对应的是[CLS]）传入classifier网络，然后进行分类任务。

## BERT的预训练任务
BERT是一个多任务模型，它的任务是由两个自监督任务组成。

### Masked Language Model（MLM）
MLM：将输入文本序列的部分（15%）单词随机Mask掉，让BERT来预测这些被Mask的词语。*（可以说是完形填空）*
>Masked Language Model（MLM）和核心思想取自Wilson Taylor在1953年发表的一篇论文《cloze procedure: A new tool for measuring readability》。所谓MLM是指在训练的时候随即从输入预料上mask掉一些单词，然后通过的上下文预测该单词，该任务非常像我们在中学时期经常做的完形填空。正如传统的语言模型算法和RNN匹配那样，MLM的这个性质和Transformer的结构是非常匹配的。

### Next Sentence Prediction（NSP）
NSP：判断两个句子是否是相邻句子。即，输入是sentence A和sentence B，经过BERT编码之后，使用CLS token的向量表示来预测两个句子是否是相邻句子。
>Next Sentence Prediction（NSP）的任务是判断句子B是否是句子A的下文。如果是的话输出’IsNext‘，否则输出’NotNext‘。训练数据的生成方式是从平行语料中随机抽取的连续两句话，其中50%保留抽取的两句话，它们符合IsNext关系，另外50%的第二句话是随机从预料中提取的，它们的关系是NotNext的。这个关系保存在[CLS]符号中。

## BERT的应用
### 特征提取
由于BERT模型可以得到输入序列所对应的所有token的向量表示，因此不仅可以使用最后一程BERT的输出连接上任务网络进行微调，还可以直接使用这些token的向量当作特征。比如，可以直接提取每一层encoder的token表示当作特征，输入现有的特定任务神经网络中进行训练。

### Pretrain + Fine tune

## 参考资料
- 基于transformers的自然语言处理(NLP)入门--在线阅读 [https://datawhalechina.github.io/learn-nlp-with-transformers/#/](https://datawhalechina.github.io/learn-nlp-with-transformers/#/)
- 李宏毅机器学习2019-ELMO,BERT,GPT [https:// www.bilibili.com/video/BV1Gb411n7dE?p=61](https://www.bilibili.com/video/BV1Gb411n7dE?p=61)
