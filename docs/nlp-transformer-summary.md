---
title: Summary Transformer课程总结
toc: true
tags: [NLP, 预训练模型, transfomer, 组队学习, 总结]
categories: [04 组队学习, 2021-09 基于transformer的NLP]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/nlp/00_bg.png?raw=true"
date: 2021-09-30 15:44:01
---
# 我的背景

第一次参加Datawhale组队学习课程，我的相关知识背景是：
- Transformer：0基础
- PyTorch：0基础
- NLP：0.1基础
- Python：0基础

作为NLP情感分析的领航员和Transformers的学员，我将从课程内容和运营两方面写一下自己的感受和想法。

# Transformer课程大纲
<img src="https://github.com/chuxiaoyu/blog_image/blob/master/nlp/transformer_xmind.png?raw=true" width="800" alt="基于Transformers的自然语言处理" align="center" />

# 课程内容方面

- 课程内容对零基础入门的人是比较友好的。比如我是第一次学习Transformer，但是图解系列很容易理解。
- 但是每个task的内容难度差别过大。比如Task02的Transformer原论文代码标注 ，和Task05 transformers源码讲解。
- 每个task的工作量不平衡，有的特别多，有的相对少。
- 第四章可以任选一个任务应用，把更多时间留给学习如何使用huggingface的transformers。（就是那个官方课程）

# 课程运营方面
- 优秀队员和优秀队长评选标准需要统一
- 补卡规则需要统一
- 逐步完善课程体系
- 小程序，用起来不是很方便
- （脑洞1）每次打卡之后马上进行作业评审和反馈（过于耗费助教精力）
- （脑洞2）给每个小组或个人安排一个期末大project，或者布置平时作业（过于耗费学员精力）
- （脑洞3）直接用一个课程管理系统进行管理（类似于canvas,雨课堂）(逐渐学院化)

# 参考资料清单（总）

Transformer在网上有很多很多教程，其中公认的、普遍性的比较好的资料如下：

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