# 05 自动程序优化

参考资料

- 英文课程主页 https://mlc.ai/summer22/ 英文课程材料 https://mlc.ai/index.html

- 中文课程主页 https://mlc.ai/summer22-zh/ 中文课程材料 https://mlc.ai/zh/index.html

- ⭐️本节代码：[ipynb](https://github.com/mlc-ai/notebooks/blob/main/5_Automatic_Program_Optimization.ipynb)

---

在过去的章节中，我们学习了如何构建元张量函数并将它们连接起来以进行端到端的模型执行。本章将讨论自动化一些流程的方法。

历史轨迹 (trace)：包含了 IRModule 在变换过程中所涉及的步骤。

随机调度变换 (Stochastic Schedule Transformation)：在我们的变换中添加一些随机元素。

随机变换搜索：使用随机变换来指定好的程序的搜索空间，使用 ``tune_tir`` API 帮助在搜索空间内搜索并找到最优的调度变换。

自动调度：Meta-Schedule 带有内置通用随机变换集合，能够适用于广泛的 TensorIR 计算。这种方法也称为自动调度 (auto-scheduling)，因为搜索空间是由系统生成的。

从 MLC 的角度来看，自动搜索是一个模块化的步骤，我们只需要用调优结果提供的新的元张量函数实现替换原始的元张量函数实现。
