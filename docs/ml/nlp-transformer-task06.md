---
title: Task06 BERT应用、训练和优化
toc: true
tags: [NLP, 预训练模型, 组队学习, transfomer, BERT, 笔记]
categories: [04 组队学习, 2021-09 基于transformer的NLP]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/nlp/00_bg.png?raw=true"
date: 2021-09-24 15:18:42
---
# Task06 BERT应用、训练和优化
*该部分的内容翻译自🤗HuggingFace官网教程第1部分（1-4章），见 [https://huggingface.co/course/chapter1](https://huggingface.co/course/chapter1)。该系列教程由3大部分共12章组成（如图），其中第1部分介绍transformers库的主要概念、模型的工作原理和使用方法、怎样在特定数据集上微调等内容。*
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/hf_1.png?raw=true" width="500" alt="" align="center" />

## 环境搭建
简单的说，有两种可以跑模型代码的方式：
1. Google Colab
2. 本地虚拟环境 `pip install transformers`

详见 [https://huggingface.co/course/chapter0?fw=pt](https://huggingface.co/course/chapter0?fw=pt)

## Transformer模型概述

### Transformers, 可以做什么？

目前可用的一些pipeline是：
- feature-extraction 获取文本的向量表示
- fill-mask 完形填空
- ner (named entity recognition) 命名实体识别
- question-answering 问答
- sentiment-analysis 情感分析
- summarization 摘要生成
- text-generation 文本生成
- translation 翻译
- zero-shot-classification 零样本分类

*pipeline: 直译管道/流水线，可以理解为流程。*

### Transformers, 如何工作？

#### Transformer简史
Transformer 架构于 2017 年 6 月推出。原始研究的重点是翻译任务。随后推出了几个有影响力的模型，包括：
- 2018 年 6 月：GPT，第一个预训练的 Transformer 模型，用于各种 NLP 任务的微调并获得最先进的结果
- 2018 年 10 月：BERT，另一个大型预训练模型，该模型旨在生成更好的句子摘要
- 2019 年 2 月：GPT-2，GPT 的改进（和更大）版本
- 2019 年 10 月：DistilBERT，BERT 的蒸馏版本，速度提高 60%，内存减轻 40%，但仍保留 BERT 97% 的性能
- 2019 年 10 月：BART 和 T5，两个使用与原始 Transformer 模型相同架构的大型预训练模型（第一个这样做）
- 2020 年 5 月，GPT-3，GPT-2 的更大版本，无需微调即可在各种任务上表现良好（称为零样本学习zero-shot learning）

大体上，它们可以分为三类：
- GPT类（又称为自回归 Transformer 模型）：只使用transformer-decoder部分
- BERT类（又称为自编码 Transformer 模型）：只使用transformer-encoder部分
- BART/T5类（又称为序列到序列 Transformer 模型）：使用Transformer-encoder-decoder部分

它们的分类、具体模型、主要应用任务如下：
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/hf_2.jpg?raw=true" width="800" alt="" align="center" />


其他需要知道的：
- Transformers是语言模型
- Transformers是大模型
- Transformers的应用通过预训练和微调两个过程

#### 名词解释：Architecture和Checkpoints
**Architecture/架构**：定义了模型的基本结构和基本运算。
**Checkpoints/检查点**：模型的某个训练状态，加载此checkpoint会加载此时的权重。训练时可以选择自动保存checkpoint。模型在训练时可以设置自动保存于某个时间点（比如模型训练了一轮epoch，更新了参数，将这个状态的模型保存下来，为一个checkpoint。） 所以每个checkpoint对应模型的一个状态，一组权重。

## 使用Transformers

### 3个处理步骤

将一些文本传递到pipeline时涉及3个主要步骤：
1. 文本被预处理为模型可以理解的格式。
2. 预处理后的输入传递给模型。
3. 模型的预测结果被后处理为人类可以理解的格式。

Pipeline将3个步骤组合在一起：预处理/Tokenizer、通过模型传递输入/Model和后处理/Post-Processing：
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/hf_3.png?raw=true" width="800" alt="" align="center" />

### Tokenizer/预处理
Tokenizer的作用：
- 将输入拆分为称为token的单词、子词/subword或符号/symbols（如标点符号）
- 将每个token映射到一个整数
- 添加可能对模型有用的其他输入

### Going Through Models/穿过模型

#### 模型实例化
```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```
在这段代码中，我们下载了在pipeline中使用的相同检查点（实际上已经缓存）并将模型实例化。

#### 模型的输出：高维向量
模型的输出向量通常有三个维度：
- Batch size: 一次处理的序列数
- Sequence length: 序列向量的长度
- Hidden size: 每个模型输入处理后的向量维度（hidden state vector）

#### Model Heads：为了处理不同的任务
Model heads:将隐藏状态的高维向量作为输入，并将它们投影到不同的维度上。它们通常由一个或几个线性层组成。
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/hf_4.png?raw=true" width="800" alt="这个图表示了Pipeline第二步在经过模型时发生的事情。" align="center" />
如上图所示，紫色代表向量，粉色代表模组，Embeddings+layers表示Transformer的架构，经过这层架构后的输出送入Model Head进行处理，从而应用到不同的下游任务。
🤗 Transformers 中有许多不同的Head架构可用，每一种架构都围绕着处理特定任务而设计。 下面列举了部分Model heads：

- *Model (retrieve the hidden states)
- *ForCausalLM
- *ForMaskedLM
- *ForMultipleChoice
- *ForQuestionAnswering
- *ForSequenceClassification
- *ForTokenClassification
- and others 🤗

### Post-processing/后处理
从模型中获得的作为输出的值本身并不一定有意义。要转换为概率，它们需要经过一个 SoftMax 层。

## 微调一个预训练模型

#### 数据处理
在本节中，我们将使用MRPC（Microsoft Research Praphrase Corpus）数据集作为示例。该DataSet由5,801对句子组成，标签指示它们是否是同义句（即两个句子是否表示相同的意思）。 我们选择它是因为它是一个小型数据集，因此可以轻松训练。

#### 从Hub上加载数据集
Hub不仅包含模型，还含有多种语言的datasets。
例如，MRPC数据集是构成 GLUE benchmark的 10 个数据集之一。GLUE（General Language Understanding Evaluation）是一个多任务的自然语言理解基准和分析平台。GLUE包含九项NLU任务，语言均为英语。GLUE九项任务涉及到自然语言推断、文本蕴含、情感分析、语义相似等多个任务。像BERT、XLNet、RoBERTa、ERINE、T5等知名模型都会在此基准上进行测试。

🤗 Datasets库提供了一个非常简单的命令来下载和缓存Hub上的dataset。 我们可以像这样下载 MRPC 数据集：
```python
>>> from datasets import load_dataset

>>> raw_datasets = load_dataset("glue", "mrpc")
>>> raw_datasets
```
输出如下：
```python
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
```
这样就得到一个DatasetDict对象，包含训练集、验证集和测试集，训练集中有3,668 个句子对，验证集中有408对，测试集中有1,725 对。每个句子对包含四个字段：'sentence1', 'sentence2', 'label'和 'idx'。

我们可以通过索引访问raw_datasets 的句子对：
```python
>>> raw_train_dataset = raw_datasets["train"]
>>> raw_train_dataset[0]
```
输出如下：
```python
{'sentence1': 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .', 
'sentence2': 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .', 
'label': 1, 
'idx': 0}
```
我们可以通过features获得数据集的字段类型：
```python
>>> raw_train_dataset.features
```
输出如下：
```python
{'sentence1': Value(dtype='string', id=None), 
'sentence2': Value(dtype='string', id=None), 
'label': ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'], names_file=None, id=None), 
'idx': Value(dtype='int32', id=None)}
```


>TIPS：
>1. 没有数据集的话首先安装一下：`pip install datasets`
>2. 这里很容易出现连接错误，解决方法如下：[https://blog.csdn.net/qq_20849045/article/details/117462846?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link](https://blog.csdn.net/qq_20849045/article/details/117462846?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link)


#### 数据集预处理
通过数据集预处理，我们将文本转换成模型能理解的向量。这个过程通过Tokenizer实现：
```python
>>> from transformers import AutoTokenizer

>>> checkpoint = "bert-base-uncased"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
>>> tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
>>> tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
```

（TODO）

### 使用Trainer API微调一个模型

#### 训练
#### 评估函数

## 补充部分
### 为什么4中用Trainer来微调模型？
### Training Arguments主要参数
### 不同模型的加载方式
### Dynamic Padding——动态填充技术

## 参考资料
- 基于transformers的自然语言处理(NLP)入门--在线阅读 [https://datawhalechina.github.io/learn-nlp-with-transformers/#/](https://datawhalechina.github.io/learn-nlp-with-transformers/#/)
- Huggingface官方教程 [https://huggingface.co/course/chapter1](https://huggingface.co/course/chapter1)