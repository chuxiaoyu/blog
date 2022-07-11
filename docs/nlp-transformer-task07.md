---
title: Task07 使用Transformers解决文本分类任务
toc: true
tags: [NLP, 预训练模型, 组队学习, transfomer, BERT, 笔记, 文本分类]
categories: [04 组队学习, 2021-09 基于transformer的NLP]
thumbnail: "https://github.com/chuxiaoyu/blog_image/blob/master/nlp/00_bg.png?raw=true"
date: 2021-09-25 16:57:45
---
# Task07 使用Transformers解决文本分类任务
*该部分的内容翻译自🤗HuggingFace/notebooks [https://github.com/huggingface/notebooks/tree/master/examples](https://github.com/huggingface/notebooks/tree/master/examples)*
*中文翻译：Datawhale/learn-nlp-with-transformers/4.1-文本分类 [Datawhale/learn-nlp-with-transformers/4.1-文本分类](https://github.com/datawhalechina/learn-nlp-with-transformers/blob/main/docs/%E7%AF%87%E7%AB%A04-%E4%BD%BF%E7%94%A8Transformers%E8%A7%A3%E5%86%B3NLP%E4%BB%BB%E5%8A%A1/4.1-%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.md)*

## 微调预训练模型进行文本分类
我们将使用 🤗 Transformers代码库中的模型来解决文本分类任务，任务来源于GLUE Benchmark.
GLUE榜单包含了9个句子级别的分类任务，分别是：
- CoLA (Corpus of Linguistic Acceptability) 鉴别一个句子是否语法正确.
- MNLI (Multi-Genre Natural Language Inference) 给定一个假设，判断另一个句子与该假设的关系：entails, contradicts 或者 unrelated。
- MRPC (Microsoft Research Paraphrase Corpus) 判断两个句子是否互为paraphrases.
- QNLI (Question-answering Natural Language Inference) 判断第2句是否包含第1句问题的答案。
- QQP (Quora Question Pairs2) 判断两个问句是否语义相同。
- RTE (Recognizing Textual Entailment)判断一个句子是否与假设成entail关系。
- SST-2 (Stanford Sentiment Treebank) 判断一个句子的情感正负向.
- STS-B (Semantic Textual Similarity Benchmark) 判断两个句子的相似性（分数为1-5分）。
- WNLI (Winograd Natural Language Inference) Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not.

对于以上任务，我们将展示如何使用简单的Dataset库加载数据集，同时使用transformer中的Trainer接口对预训练模型进行微调。
```python
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
```

This notebook is built to run on any of the tasks in the list above, with any model checkpoint from the Model Hub as long as that model has a version with a classification head. Depending on you model and the GPU you are using, you might need to adjust the batch size to avoid out-of-memory errors. Set those three parameters, then the rest of the notebook should run smoothly:
本notebook理论上可以使用各种各样的transformer模型（模型面板），解决任何文本分类分类任务。如果您所处理的任务有所不同，大概率只需要很小的改动便可以使用本notebook进行处理。同时，您应该根据您的GPU显存来调整微调训练所需要的btach size大小，避免显存溢出。
```python
task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
```

## 加载数据集
We will use the 🤗 Datasets library to download the data and get the metric we need to use for evaluation (to compare our model to the benchmark). This can be easily done with the functions `load_dataset` and `load_metric`.
我们将会使用🤗 Datasets库来加载数据和对应的评测方式。数据加载和评测方式加载只需要简单使用load_dataset和load_metric即可。
```pyton
from datasets import load_dataset, load_metric
```

Apart from mnli-mm being a special code, we can directly pass our task name to those functions. `load_dataset` will cache the dataset to avoid downloading it again the next time you run this cell.
除了mnli-mm以外，其他任务都可以直接通过任务名字进行加载。数据加载之后会自动缓存。
```python
actual_task = "mnli" if task == "mnli-mm" else task
dataset = load_dataset("glue", actual_task)
metric = load_metric('glue', actual_task)
```
*上节讲过，这里，最好手动下载glue.py和gule_metric.py，不下载到本地的话，容易出现连接错误。*

The dataset object itself is DatasetDict, which contains one key for the training, validation and test set (with more keys for the mismatched validation and test set in the special case of mnli).
这个datasets对象本身是一种DatasetDict数据结构.对于训练集、验证集和测试集，只需要使用对应的key（train，validation，test）即可得到相应的数据。
```python
>>> dataset
DatasetDict({
    train: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 8551
    })
    validation: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1043
    })
    test: Dataset({
        features: ['sentence', 'label', 'idx'],
        num_rows: 1063
    })
})
```

```python
>>> dataset["train"][0]
{'sentence': "Our friends won't buy this analysis, let alone the next one we propose.",
'label': 1, 
'idx': 0}
```

To get a sense of what the data looks like, the following function will show some examples picked randomly in the dataset.
为了能够进一步理解数据长什么样子，下面的函数将从数据集里随机选择几个例子进行展示。
```python
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))

```
```python
show_random_elements(dataset["train"])
```

The metric is an instance of datasets.Metric:
```python
pass
```

You can call its `compute` method with your predictions and labels directly and it will return a dictionary with the metric(s) value:
直接调用metric的compute方法，传入labels和predictions即可得到metric的值：
```python
import numpy as np

fake_preds = np.random.randint(0, 2, size=(64,))
fake_labels = np.random.randint(0, 2, size=(64,))
metric.compute(predictions=fake_preds, references=fake_labels)
```
Note that load_metric has loaded the proper metric associated to your task, which is:
每一个文本分类任务所对应的metic有所不同，具体如下:
- for CoLA: Matthews Correlation Coefficient
- for MNLI (matched or mismatched): Accuracy
- for MRPC: Accuracy and F1 score
- for QNLI: Accuracy
- for QQP: Accuracy and F1 score
- for RTE: Accuracy
- for SST-2: Accuracy
- for STS-B: Pearson Correlation Coefficient and Spearman's_Rank_Correlation_Coefficient
- for WNLI: Accuracy

so the metric object only computes the one(s) needed for your task.

## 数据预处理
Before we can feed those texts to our model, we need to preprocess them. This is done by a 🤗 Transformers `Tokenizer` which will (as the name indicates) tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained vocabulary) and put it in a format the model expects, as well as generate the other inputs that model requires.
在将数据喂入模型之前，我们需要对数据进行预处理。预处理的工具叫Tokenizer。Tokenizer首先对输入进行tokenize，然后将tokens转化为预模型中需要对应的token ID，再转化为模型需要的输入格式。

To do all of this, we instantiate our tokenizer with the `AutoTokenizer.from_pretrained` method, which will ensure:
- we get a tokenizer that corresponds to the model architecture we want to use,
- we download the vocabulary used when pretraining this specific checkpoint.

为了达到数据预处理的目的，我们使用AutoTokenizer.from_pretrained方法实例化我们的tokenizer，这样可以确保：
- 我们得到一个与预训练模型一一对应的tokenizer。
- 使用指定的模型checkpoint对应的tokenizer的时候，我们也下载了模型需要的词表库vocabulary，准确来说是tokens vocabulary。

That vocabulary will be cached, so it's not downloaded again the next time we run the cell.
这个被下载的tokens vocabulary会被缓存起来，从而再次使用的时候不会重新下载。
```python
from transformers import AutoTokenizer
    
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
```

You can directly call this tokenizer on one sentence or a pair of sentences:
```python
tokenizer("Hello, this one sentence!", "And this sentence goes with it.")
```
输出为：
pass

To preprocess our dataset, we will thus need the names of the columns containing the sentence(s). The following dictionary keeps track of the correspondence task to column names:
```python
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
```

We can double check it does work on our current dataset:
```python
sentence1_key, sentence2_key = task_to_keys[task]
if sentence2_key is None:
    print(f"Sentence: {dataset['train'][0][sentence1_key]}")
else:
    print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
    print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")
```
输出为：
```
Sentence: Our friends won't buy this analysis, let alone the next one we propose.
```

We can them write the function that will preprocess our samples. We just feed them to the `tokenizer` with the argument `truncation=True`. This will ensure that an input longer that what the model selected can handle will be truncated to the maximum length accepted by the model.
```python
def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
```
```python
>>> preprocess_function(dataset['train'][:5])
{'input_ids': [[101, 2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012, 102], [101, 2028, 2062, 18404, 2236, 3989, 1998, 1045, 1005, 1049, 3228, 2039, 1012, 102], [101, 2028, 2062, 18404, 2236, 3989, 2030, 1045, 1005, 1049, 3228, 2039, 1012, 102], [101, 1996, 2062, 2057, 2817, 16025, 1010, 1996, 13675, 16103, 2121, 2027, 2131, 1012, 102], [101, 2154, 2011, 2154, 1996, 8866, 2024, 2893, 14163, 8024, 3771, 1012, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
```

To apply this function on all the sentences (or pairs of sentences) in our dataset, we just use the `map` method of our `dataset` object we created earlier. This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command.
接下来对数据集datasets里面的所有样本进行预处理，处理的方式是使用map函数，将预处理函数prepare_train_features应用到（map)所有样本上。
```python
encoded_dataset = dataset.map(preprocess_function, batched=True)
```

## 微调模型
Now that our data is ready, we can download the pretrained model and fine-tune it. Since all our tasks are about sentence classification, we use the AutoModelForSequenceClassification class. Like with the tokenizer, the from_pretrained method will download and cache the model for us. The only thing we have to specify is the number of labels for our problem (which is always 2, except for STS-B which is a regression problem and MNLI where we have 3 labels):
既然数据已经准备好了，现在我们需要下载并加载我们的预训练模型，然后微调预训练模型。既然我们是做seq2seq任务，那么我们需要一个能解决这个任务的模型类。我们使用AutoModelForSequenceClassification 这个类。和tokenizer相似，from_pretrained方法同样可以帮助我们下载并加载模型，同时也会对模型进行缓存，就不会重复下载模型啦。
```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
```
输出为：
```
Some weights of the model checkpoint at distilbert-base-uncased were not used when initializing DistilBertForSequenceClassification: ['vocab_transform.bias', 'vocab_projector.weight', 'vocab_layer_norm.bias', 'vocab_layer_norm.weight', 'vocab_projector.bias', 'vocab_transform.weight']
- This IS expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing DistilBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```
The warning is telling us we are throwing away some weights (the vocab_transform and vocab_layer_norm layers) and randomly initializing some other (the pre_classifier and classifier layers). This is absolutely normal in this case, because we are removing the head used to pretrain the model on a masked language modeling objective and replacing it with a new head for which we don't have pretrained weights, so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.
由于我们微调的任务是文本分类任务，而我们加载的是预训练的语言模型，所以会提示我们加载模型的时候扔掉了一些不匹配的神经网络参数（比如：预训练语言模型的神经网络head被扔掉了，同时随机初始化了文本分类的神经网络head）。

To instantiate a `Trainer`, we will need to define two more things. The most important is the `TrainingArguments`, which is a class that contains all the attributes to customize the training. It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:
为了能够得到一个Trainer训练工具，我们还需要3个要素，其中最重要的是训练的设定/参数 TrainingArguments。这个训练设定包含了能够定义训练过程的所有属性。
```python
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    "test-glue",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
    push_to_hub_model_id=f"{model_name}-finetuned-{task}",
)
```
Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the `batch_size` defined at the top of the notebook and customize the number of epochs for training, as well as the weight decay. Since the best model might not be the one at the end of training, we ask the `Trainer` to load the best model it saved (according to `metric_name`) at the end of training.
上面evaluation_strategy = "epoch"参数告诉训练代码：我们每个epcoh会做一次验证评估。
上面batch_size在这个notebook之前定义好了。

The last two arguments are to setup everything so we can push the model to the `Hub` at the end of training. Remove the two of them if you didn't follow the installation steps at the top of the notebook, otherwise you can change the value of `push_to_hub_model_id` to something you would prefer.
*(后面需要连接到hub客户端，太麻烦，所以先设为False)*

The last thing to define for our `Trainer` is how to compute the metrics from the predictions. We need to define a function for this, which will just use the `metric` we loaded earlier, the only preprocessing we have to do is to take the argmax of our predicted logits (our just squeeze the last axis in the case of STS-B):
最后，由于不同的任务需要不同的评测指标，我们定一个函数来根据任务名字得到评价方法:
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)
```

Then we just need to pass all of this along with our datasets to the Trainer:
```python
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```
>BUG:
>ValueError: You must login to the Hugging Face hub on this computer by typing `transformers-cli login` and entering your credentials to use `use_auth_token=True`. Alternatively, you can pass your own token as the `use_auth_token` argument.
>把args里面的该项参数改为False   `push_to_hub=False,`。

We can now finetune our model by just calling the `train` method:
```python
trainer.train()
```
输出为：
```python
pass
```

We can check with the `evaluate` method that our `Trainer` did reload the best model properly (if it was not the last one):
```python
trainer.evaluate()
```
输出为：
```python
pass
```

## 超参搜索
The Trainer supports hyperparameter search using optuna or Ray Tune. 
```shell
pip install optuna
pip install ray[tune]
```

During hyperparameter search, the Trainer will run several trainings, so it needs to have the model defined via a function (so it can be reinitialized at each new run) instead of just having it passed. We jsut use the same function as before:
超参搜索时，Trainer将会返回多个训练好的模型，所以需要传入一个定义好的模型从而让Trainer可以不断重新初始化该传入的模型：
```python
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
```

And we can instantiate our Trainer like before:
```python
trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

The method we call this time is `hyperparameter_search`. Note that it can take a long time to run on the full dataset for some of the tasks. You can try to find some good hyperparameter on a portion of the training dataset by replacing the `train_dataset` line above by:
`train_dataset = encoded_dataset["train"].shard(index=1, num_shards=10) `
for 1/10th of the dataset. Then you can run a full training on the best hyperparameters picked by the search.
调用方法hyperparameter_search。注意，这个过程可能很久，我们可以先用部分数据集进行超参搜索，再进行全量训练。 比如使用1/10的数据进行搜索：
```python
best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")
```

The hyperparameter_search method returns a `BestRun` objects, which contains the value of the objective maximized (by default the sum of all metrics) and the hyperparameters it used for that run.
hyperparameter_search会返回效果最好的模型相关的参数：
```python
>>> best_run

```

To reproduce the best training, just set the hyperparameters in your TrainingArgument before creating a Trainer:
将Trainner设置为搜索到的最好参数，进行训练：
```python
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()
```


## 参考资料
- HuggingFace/transfomers/BERT [https://huggingface.co/transformers/model_doc/bert.html#](https://huggingface.co/transformers/model_doc/bert.html#)
- 基于transformers的自然语言处理(NLP)入门--在线阅读 [https://datawhalechina.github.io/learn-nlp-with-transformers/#/](https://datawhalechina.github.io/learn-nlp-with-transformers/#/)
