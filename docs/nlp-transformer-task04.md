---
title: Task04 学习GPT
toc: true
tags: [NLP, 预训练模型, 组队学习, GPT, 笔记]
categories: [04 组队学习, 2021-09 基于transformer的NLP]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/nlp/00_bg.png?raw=true"
date: 2021-09-19 20:02:17
---
# Task04 学习GPT
## 从语言模型说起

### 自编码语言模型（auto-encoder）

自编码语言模型通过随机Mask输入的部分单词，然后预训练的目标是预测被Mask的单词，不仅可以融入上文信息，还可以自然的融入下文信息。ex. BERT.
- 优点：自然地融入双向语言模型，同时看到被预测单词的上文和下文
- 缺点：训练和预测不一致。训练的时候输入引入了[Mask]标记，但是在预测阶段往往没有这个[Mask]标记，导致预训练阶段和Fine-tuning阶段不一致。

### 自回归语言模型（auto-regressive）

语言模型根据输入句子的一部分文本来预测下一个词。ex. GPT-2
- 优点：对于生成类的NLP任务，比如文本摘要，机器翻译等，从左向右的生成内容，天然和自回归语言模型契合。
- 缺点：由于一般是从左到右（当然也可能从右到左），所以只能利用上文或者下文的信息，不能同时利用上文和下文的信息。

## Transformer, BERT, GPT-2的关系
Transformer的Encoder进化成了BERT，Decoder进化成了GPT2。

如果要使用Transformer来解决语言模型任务，并不需要完整的Encoder部分和Decoder部分，于是在原始Transformer之后的许多研究工作中，人们尝试只使用Transformer Encoder或者Decoder进行预训练。比如BERT只使用了Encoder部分进行masked language model（自编码）训练，GPT-2便是只使用了Decoder部分进行自回归（auto regressive）语言模型训练。

## GPT-2概述

### 模型的输入

输入的处理分为两步：token embedding + position encoding。即:
1. 在嵌入矩阵中查找输入的单词的对应的embedding向量
2. 融入位置编码

### Decoder层
每一层decoder的组成：Masked Self-Attention + Feed Forward Neural Network

Self-Attention所做的事情是：它通过对句子片段中每个词的相关性打分，并将这些词的表示向量根据相关性加权求和，从而让模型能够将词和其他相关词向量的信息融合起来。

Masked Self-Attention做的是：将mask位置对应的的attention score变成一个非常小的数字或者0，让其他单词再self attention的时候（加权求和的时候）不考虑这些单词。

### 模型的输出

当模型顶部的Decoder层产生输出向量时，模型会将这个向量乘以一个巨大的嵌入矩阵（vocab size x embedding size）来计算该向量和所有单词embedding向量的相关得分。这个相乘的结果，被解释为模型词汇表中每个词的分数，经过softmax之后被转换成概率。

我们可以选择最高分数的 token（top_k=1），也可以同时考虑其他词（top k）。假设每个位置输出k个token，假设总共输出n个token，那么基于n个单词的联合概率选择的输出序列会更好。

模型完成一次迭代，输出一个单词。模型会继续迭代，直到所有的单词都已经生成，或者直到输出了表示句子末尾的token。

## 关于Self-Attention, Masked Self-Attention

### Self-Attention
Self-Attention 主要通过 3 个步骤来实现：

1. 为每个路径创建 Query、Key、Value 矩阵。
2. 对于每个输入的token，使用它的Query向量为所有其他的Key向量进行打分。
3. 将 Value 向量乘以它们对应的分数后求和。

### Masked Self-Attention
在Self-Attention的第2步，把未来的 token 评分设置为0，因此模型不能看到未来的词。

这个屏蔽（masking）经常用一个矩阵来实现，称为 attention mask矩阵。
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/mask_1.jpg?raw=true" width="600" alt="" align="center" />
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/mask_2.jpg?raw=true" width="600" alt="" align="center" />

### GPT-2中的Self-Attention
(skip)

## 自回归语言模型的应用
应用在下游并取得不错效果的NLP任务有：机器翻译、摘要生成、音乐生成。*（可见，主要是跟预训练任务相似的生成类任务。）*


## 参考资料
- 基于transformers的自然语言处理(NLP)入门--在线阅读 [https://datawhalechina.github.io/learn-nlp-with-transformers/#/](https://datawhalechina.github.io/learn-nlp-with-transformers/#/)