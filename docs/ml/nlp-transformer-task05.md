---
title: Task05 编写BERT模型
toc: true
tags: [NLP, 预训练模型, 组队学习, transfomer, BERT, 笔记]
categories: [04 组队学习, 2021-09 基于transformer的NLP]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/nlp/00_bg.png?raw=true"
date: 2021-09-21 03:01:35
---
# Task05 编写BERT模型
## Overview
本部分是BERT源码的解读，来自HuggingFace/transfomers/BERT[1]。
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/bert_1.png?raw=true" width="600" alt="" align="center" />

如图所示，代码结构和作用如下：

- BertTokenizer 预处理和切词
- BertModel
    - BertEmbeddings 词嵌入
    - BertEncoder
        + BertAttention 注意力机制
        + BertIntermediate 全连接和激活函数
        + BertOutput 全连接、残差链接和正则化
    - BertPooler 取出[CLS]对应的向量，然后通过全连接层和激活函数后输出结果

## BERT的实现
### BertConfig
~~~python
classtransformers.BertConfig(vocab_size=30522, hidden_size=768, 
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, 
    hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1
    , max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, 
    layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False, 
    position_embedding_type='absolute', use_cache=True, classifier_dropout=None,
     **kwargs)
~~~
这是存储BertModel（Torch.nn.Module的子类）或TFBertModel（tf.keras.Model的子类）配置的配置类。它用于根据指定的参数来实例化BERT模型，定义模型架构。

配置对象从PretrainedConfig继承，可用于控制模型输出。


## 参考资料
- HuggingFace/transfomers/BERT [https://huggingface.co/transformers/model_doc/bert.html#](https://huggingface.co/transformers/model_doc/bert.html#)
- 基于transformers的自然语言处理(NLP)入门--在线阅读 [https://datawhalechina.github.io/learn-nlp-with-transformers/#/](https://datawhalechina.github.io/learn-nlp-with-transformers/#/)
