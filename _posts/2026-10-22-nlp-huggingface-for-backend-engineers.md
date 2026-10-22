---
title: "后端工程师的 NLP 与 Hugging Face 入门"
description: "面向业务后端开发者，用 14 张图理解分词、词向量、RNN、Transformer 与预训练，跑通评论分类、翻译和 Hugging Face 微调流程。附完整示例与下载包。"
date: 2026-10-22 10:00:00 +0800
categories: [人工智能, 自然语言处理]
tags: [NLP, HuggingFace, Transformer, BERT, pytorch, python, 后端开发]
math: true
toc: true
media_subpath: /assets/img/posts/nlp-huggingface
---

你可能已经写过这样的业务逻辑：用户提交一条评价，服务把它存进数据库，再按商品、时间和评分查询。现在，产品希望系统继续判断：这条评价是在夸物流，还是在抱怨售后？能不能自动归类工单、提取订单号，再生成一句合适的回复？

这时，数据库里的 `text` 字段就不只是需要保存的字符串了。我们希望程序从文字中提取信息，并据此作出判断。**自然语言处理，也就是 NLP，研究的正是这类问题。**

这篇文章沿着两个场景展开：**评论情感分类**和**中英翻译**。前者帮助我们理解“读完一句话，作出判断”，后者帮助我们理解“读完输入，生成另一段话”。从分词、词向量到 Transformer，再到 Hugging Face 的实际用法，你会看到每一种技术解决了上一种方法的什么困难。

读完之后，你应当能够解释一段文本如何变成模型输入，分清 BERT、GPT、T5 的主要用途，并跑通一个小型的模型训练与推理流程。你不需要预先学过 AI，但最好能够阅读 Python 函数、列表、字典和类。

正文先建立直觉，再介绍必要的公式。配套代码默认使用 CPU，教学数据随文件提供；首次加载预训练 BERT 需要联网下载数百 MB 文件。小数据示例用于理解流程，不能用来证明模型具备生产质量。

<!-- download-package:start -->
<a id="download"></a>

[下载完整学习包（ZIP）]({{ '/assets/downloads/nlp-huggingface-demo.zip' | relative_url }})。解压后进入 `nlp-huggingface-demo/`，再运行本文命令。下载包包含离线 Markdown 正文、14 张教学图、9 个 Python 源文件、教学数据、固定版本依赖及核验说明；其中 `common.py` 是共享模块，其余 8 个文件是运行入口。
<!-- download-package:end -->

## 阅读路线

第一次阅读可以先按顺序读概念。如果想先看到运行结果，也可以在第 4 节之后先读第 11—14 节，再回到第 5—10 节理解模型内部。

- [1 从业务问题认识 NLP](#s1)
- [2 先准备能运行示例的环境](#s2)
- [3 文本表示的第一步是分词](#s3)
- [4 从编号到有用的向量](#s4)
- [5 RNN 如何读取一段话](#s5)
- [6 LSTM 和 GRU 如何选择记忆](#s6)
- [7 Seq2Seq 如何生成另一段话](#s7)
- [8 Attention 如何按需读取上下文](#s8)
- [9 Transformer 如何组织这些计算](#s9)
- [10 预训练如何复用语言知识](#s10)
- [11 认识 Hugging Face 的分工](#s11)
- [12 加载模型并读懂输入输出](#s12)
- [13 用 Datasets 准备训练数据](#s13)
- [14 完成一次评论分类微调](#s14)
- [15 从教学示例走向业务服务](#s15)
- [附录 公式、API 与术语速查](#appendix)

<a id="s1"></a>

## 1 从业务问题认识 NLP

### 1.1 NLP 的输入输出是什么

自然语言是人日常使用的语言。它不像 JSON Schema 那样有严格、统一的结构：同一件事可以有多种说法，一句话还可能包含省略、反讽和领域术语。

例如，“等了三天终于到了”和“到货速度真是感人”，都可能在表达对物流的不满。只匹配“慢”这个字，就会漏掉它们。

NLP 的常见任务可以按输出形式理解：

| 任务 | 输入例子 | 输出例子 | 业务用途 |
| --- | --- | --- | --- |
| 文本分类 | 一条用户评价 | 正面或负面 | 评价分析、垃圾内容识别、工单路由 |
| 序列标注 | 一段报修描述 | 每个词或字的标签 | 找出人名、地址、商品名称 |
| 信息抽取 | 合同正文，或正文加一个问题 | 日期、金额、关系或答案片段 | 把非结构化内容转成字段 |
| 文本生成 | 对话历史或提示 | 一段新文本 | 客服回复、写作、代码生成 |
| 文本转换 | 中文句子或长文章 | 英文译文或摘要 | 翻译、改写、摘要 |

这些分类并不完全互斥。**命名实体识别（NER）**既是一种序列标注任务，也可以是信息抽取的一部分；摘要既属于文本转换，也常通过生成来实现。

同样，抽取式问答和生成式问答需要区分。前者从给定段落中定位答案的起止位置，后者生成答案文字；能生成回答，不代表答案一定来自原文。

### 1.2 从规则到预训练模型

NLP 的发展可以看成几种方法逐步补充彼此的能力，而不是旧方法全部失效。

最早的规则系统依赖人工编写词典和匹配规则。早期机器翻译演示，以及通过模式匹配进行对话的 ELIZA，体现了这种思路。规则容易解释，也适合订单号、固定错误码等明确格式，但规则数量会随表达方式增加。

统计语言模型开始利用文本出现的频率。**N-gram** 假设一个词主要依赖前面的有限几个词。例如，Bigram 用前一个词预测当前词，Trigram 使用前两个词。隐马尔可夫模型（HMM）把词性等不可直接观察的状态纳入概率模型；最大熵模型则可以组合多种人工设计的特征。

随后，词袋特征配合逻辑回归、支持向量机（SVM）、决策树等方法，成为文本分类的重要方案。条件随机场（CRF）则常用于序列标注，在预测标签时考虑标签之间的依赖。

深度学习进一步让模型从数据中学习表示。RNN、LSTM、GRU 通过状态传递处理顺序；Transformer 用注意力建立位置之间的联系；预训练则让不同业务可以复用已经学到的语言规律。

对后端开发者来说，最值得关注的变化是：过去主要通过代码写规则，现在还需要通过数据训练参数。但**规则、数据质量、模型和业务流程仍然需要一起设计**。

### 1.3 不要跳过简单基线

**词袋模型（Bag of Words）**统计一段文本里各个词出现了多少次。它不保存完整语序，所以“客服帮助用户”和“用户帮助客服”会得到相同的词频表示。

N-gram 特征可以保留一些局部顺序。例如，把“没有 回复”作为一个二元特征，就比只看到“没有”和“回复”更有用。这里的 N-gram 是特征构造方式；前面提到的 N-gram 语言模型则是用这些局部上下文估计条件概率，二者有关联，但用途不同。

另一个常见特征是 **TF-IDF**。TF 表示词在当前文档中的频率；IDF 降低到处都出现的词的权重，让更有区分力的词得到更多关注。常见形式是 `TF × log(文档总数 / 包含该词的文档数)`，实际库通常还会平滑和归一化。

对于类别固定、数据量不大、延迟要求较高的业务，TF-IDF 加线性分类器值得先跑一次。配套的 [bow_baseline.py](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/examples/bow_baseline.py) 使用中文字符的一元、二元特征，避免引入额外分词误差。后续模型的收益，应当相对于这样的基线来评估。

<a id="s2"></a>

## 2 先准备能运行示例的环境

在解压后的文件夹中运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
python examples/bow_baseline.py
```

Windows PowerShell 的激活命令是 `.venv\Scripts\Activate.ps1`。如果使用 Conda，也可以执行 `conda create -n nlp-learning python=3.12`，激活后安装同一份依赖。

本文示例锁定 Python 3.12 对应的一组依赖，包括 PyTorch 2.6.0、Transformers 4.57.1、Datasets 4.2.0、Gensim 4.4.0 和 jieba 0.42.1。锁定版本是为了便于复现，不表示这些是最新版本。完整运行说明见 [运行说明](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/README.md)。

先认识后面会反复出现的几个词：

| 术语 | 在本文中的含义 |
| --- | --- |
| 样本与标签 | 一条评价是样本；人工确定的正面、负面是标签 |
| 参数或权重 | 模型在训练过程中调整的数值，例如矩阵中的元素 |
| 张量 Tensor | 多维数值数组；向量是一维数组，矩阵是二维数组，PyTorch 张量还支持 GPU 计算 |
| 前向传播 | 输入经过模型计算，得到输出 |
| 损失 Loss | 衡量模型预测与目标之间差异的数值 |
| 反向传播 | 计算损失对参数的梯度，指出怎样调整参数 |
| 优化器 | 根据梯度更新参数的算法，例如 Adam、AdamW |
| Batch 与 Epoch | 一次一起处理的样本称为一个 batch；完整遍历训练集一次称为一个 epoch |
| 推理 Inference | 使用已有参数处理新输入，通常不更新参数 |

PyTorch 提供张量、自动求导和神经网络层；jieba 处理传统中文分词；Gensim 用于词向量；Transformers 和 Datasets 负责预训练模型及数据处理。NLTK 可用于文本处理和 BLEU 评估，TensorBoard 用来观察训练曲线，tqdm 显示进度，Jupyter 适合交互实验。后面几个工具按需要使用即可。

初学时先用 CPU 跑通小例子。如果以后使用 NVIDIA GPU，应按操作系统、驱动和 PyTorch 支持情况选择安装包。`nvidia-smi` 显示的 CUDA 信息不能简单当作本机已经安装了同版本 CUDA Toolkit，更不要把某一个 CUDA 安装命令原样套到所有机器上。具体安装组合以 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)为准。

<a id="s3"></a>

## 3 文本表示的第一步是分词

### 3.1 模型首先需要一串编号

对程序而言，“物流太慢了”是一串字符；对神经网络而言，它最终需要变成可以参与矩阵运算的数值。

常见的处理链路是：

![文本经过分词、编号、嵌入和上下文模型，最后得到任务输出](01-text-to-model.png){: width="1600" height="720" }

**分词器（Tokenizer）**把文本拆成若干 **Token**。Token 是模型处理文本的基本单位，可能是一个汉字、一个英文词、一个词片段，甚至是字节对应的单位。它不保证具有独立语义，也不一定等于人理解的“一个词”。

**词表（Vocabulary）**保存模型能够识别的 Token，并为每个 Token 分配一个整数 ID。可以把它理解成字典：`token → id`。ID 只是索引，`退款=20`、`发货=21` 不意味着它们在语义上更接近。

### 3.2 词、字符和子词怎么选

![词级、字符级和子词级分词的对比](02-tokenization.png){: width="1600" height="760" }

词级分词比较直观，但词表不可能预先收录所有新词。输入出现词表之外的词时，就遇到了 **OOV（Out of Vocabulary，未登录词）**，通常只能映射成 `<unk>`，丢失部分信息。

字符级分词的词表较小，但句子会变长，模型需要组合更多位置才能形成词义。它也不是绝对没有 OOV：未收录的字符仍可能无法表示。字节级方案在覆盖范围上有不同的设计。

子词分词在两者之间折中：常见词可以整体表示，罕见词拆成更小的片段。常见算法包括：

- **BPE**：从基础字符或字节单位开始，反复合并高频相邻片段；使用时应用已学到的合并规则。
- **WordPiece**：根据语料学习子词词表，采用不同于基础 BPE 的构造目标，BERT 使用这一类方法。
- **Unigram**：从候选子词集合出发，根据概率模型逐步精简词表，并选择合适的切分。

这些片段是从数据中学习出来的，不必刚好对应词根、前缀或后缀。中文同样可以按字、词或子词处理，不能把“中文分词”只理解为查词典。相关算法可继续阅读 [Hugging Face 分词课程](https://huggingface.co/learn/llm-course/chapter6/5)。

### 3.3 jieba 适合怎样的工作

jieba 常用于搜索、关键词统计和传统 NLP 预处理。精确模式尝试得到一种切分；全模式扫描可能成词的片段；搜索模式进一步拆开长词，便于建立检索索引。

下面的例子可以直接运行：

```python
import jieba

text = "小林毕业于北京大学计算机系"
print(jieba.lcut(text))
print(jieba.lcut(text, cut_all=True))
print(jieba.lcut_for_search(text))

jieba.add_word("订单履约")
print(jieba.lcut("订单履约出现异常"))
```

`cut` 返回可迭代结果，`lcut` 直接返回列表。业务词汇较多时，可以使用 `load_userdict()` 加载 UTF-8 词典，每行按“词语、可选词频、可选词性”的顺序填写；也可以通过 `add_word()`、`del_word()` 动态调整。三种模式和自定义词典的完整示例见 [tokenize_word2vec.py](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/examples/tokenize_word2vec.py)，接口说明见 [jieba 官方项目](https://github.com/fxsjy/jieba)。

HanLP 也是传统中文处理工具之一；Hugging Face Tokenizers、SentencePiece 和 tiktoken 则经常出现在现代模型的文本编码链路中。

**使用预训练模型时，通常直接把原始文本交给模型配套的 Tokenizer。** 不要先随意用 jieba 切词，再用空格拼回去。训练时和推理时的切分方式、词表、特殊标记不一致，即使字符串仍然“看得懂”，模型接收到的输入也已经变了。

<a id="s4"></a>

## 4 从编号到有用的向量

### 4.1 One-hot 为什么不够

假设词表只有“物流、配送、退款”三个词，One-hot 编码分别是：

```text
物流 → [1, 0, 0]
配送 → [0, 1, 0]
退款 → [0, 0, 1]
```

这种表示可以区分身份，却没有表达“物流”和“配送”的关系。词表有十万个词时，每个向量还会有十万个位置，绝大部分都是 0。

**Embedding（嵌入）**改用较短的稠密向量表示每个 Token。假设词表大小为 `V`，向量维度为 `D`，嵌入层就是一张形状为 `[V, D]` 的可学习矩阵。输入 ID 为 20，就取出第 20 行。

数学上，One-hot 向量乘这张矩阵与直接查表等价。因此实际程序通常传入整数 ID，不需要真的创建巨大的 One-hot 数组。

### 4.2 Word2Vec 为什么能学到词的关系

Word2Vec 的直觉是：**经常出现在相似上下文里的词，可能具有相近的用法。** 它把“学词义”转成可以计算损失的预测任务。

![CBOW 与 Skip-gram 的预测方向，以及静态词向量的局限](03-word2vec.png){: width="1600" height="800" }

以“订单 退款 成功”为例，把窗口大小设为 1：

- **Skip-gram**用“退款”预测周围的“订单”和“成功”，构造两个中心词与上下文词的训练对。
- **CBOW**取“订单”和“成功”的向量，通常求和或平均，再预测中间的“退款”。

训练流程可以理解为：查出输入词向量，通过另一组权重为候选词打分，用 Softmax 把得分转换为概率，再根据正确词计算交叉熵损失。反向传播同时更新相关参数，词向量因此逐渐包含对预测有帮助的信息。

Softmax 会让输出非负、总和为 1；交叉熵则会在正确答案概率过低时给出较大的损失。实际 Word2Vec 常使用负采样或层次 Softmax 降低大词表计算成本，不必每次计算所有候选词。Gensim 提供了这些训练选项，见 [Word2Vec 文档](https://radimrehurek.com/gensim/models/word2vec.html)。

最小训练示例：

```python
from gensim.models import Word2Vec

sentences = [["物流", "很快"], ["配送", "很快"],
             ["物流", "及时"], ["配送", "及时"]]
model = Word2Vec(sentences, vector_size=16, window=2,
                 min_count=1, sg=1, sample=0, workers=1, seed=42, epochs=20)
print(model.wv["物流"].shape)
print(model.wv.most_similar("物流", topn=2))
```

`sg=1` 表示 Skip-gram，`sg=0` 表示 CBOW；`window` 控制上下文窗口，`min_count` 过滤低频词。这里特意设置 `sample=0`，关闭高频词下采样：只有四句话时，默认下采样可能丢掉几乎所有可用于训练的上下文，出现“训练结束了，向量却没有更新”的情况。真实大语料中，下采样可以减少高频词的干扰，不应因为这个小例子就一律关闭。即使参数确实更新，这点语料也不能学出可靠词义。

可以通过 `Word2Vec.save/load` 保存、恢复可继续训练的模型；只需要向量时，可使用 `model.wv`，以及 `KeyedVectors.load_word2vec_format()` 加载公开的 Word2Vec 格式文件。常见文本格式第一行是“词数 向量维度”，后面每行是一个词及其数值。`vector_size`、`similarity()`、`most_similar()` 分别帮助查看维度、比较余弦相似度和寻找近邻。

余弦相似度比较向量方向，取值在 −1 到 1 之间。但它不是“词义相同的概率”，反义词也可能因为上下文相似而靠得很近。

### 4.3 怎么把词向量用于下游模型

嵌入层可以随机初始化，也可以用已有词向量初始化。后一种方式相当于给模型一个已有的起点，但是否有帮助，还取决于语料领域是否匹配。

PyTorch 提供 `nn.Embedding.from_pretrained(matrix, freeze=False, padding_idx=0)`：`freeze=False` 允许继续训练；`freeze=True` 保持向量不变。**构造矩阵时必须按自己的词表 ID 排列行。** 不能把下载文件的行顺序直接当成本地词表顺序。不存在于公开向量中的词要设置回退向量，`<pad>` 行也应单独处理。

### 4.4 静态词向量还缺少什么

“苹果很甜”和“苹果发布新手机”里的“苹果”含义不同，但 Word2Vec 会给它同一个向量。

**上下文表示**允许同一个词在不同句子里得到不同向量。ELMo 使用双向语言模型学习上下文表示，是理解这一变化的一个重要例子；BERT 则通过 Transformer 编码器实现深层的双向上下文建模。

这里要分清两层：模型入口的 Token Embedding 通常仍是查表得到的静态向量；经过后续上下文模型计算后，每个位置的隐藏表示才会随语境变化。

<a id="s5"></a>

## 5 RNN 如何读取一段话

### 5.1 把它理解成带状态的循环

词向量描述词本身，但“服务好”和“服务不好”还需要结合顺序。**RNN（Recurrent Neural Network，循环神经网络）**逐个读取 Token，并持续更新一个隐藏状态。

如果用业务代码作类比，它像一个不断执行的状态更新函数：

```text
新状态 = update(旧状态, 当前输入)
```

最基础的 RNN 可以写成：

$$
h_t=\tanh(W_x x_t+W_h h_{t-1}+b)
$$

`x_t` 是第 t 个词的向量，`h_t` 是读到这里的隐藏状态，`W_x`、`W_h` 和 `b` 是需要训练的参数。`tanh` 是一个非线性函数，将数值压到 −1 与 1 之间。所有时间步共用同一套参数，所以模型可以处理不同长度的句子。

![RNN 的状态传递以及 LSTM、GRU 的记忆方式](04-rnn-gates.png){: width="1600" height="940" }

### 5.2 多层和双向分别改变什么

多层 RNN 把第一层输出的整个序列交给第二层，增加表示能力。它增加的是深度，不是让句子变长。

双向 RNN 使用两套参数：一套从左往右读，一套从右往左读，再组合两个方向的表示。对于整句已经存在的评论分类、实体识别，这很自然。但在线逐词生成时，未来的词还不知道，不能直接把目标序列的未来信息交给双向网络。

多层与双向可以同时使用。设置 `batch_first=True` 后，常见形状是：

| 数据 | 形状 | 含义 |
| --- | --- | --- |
| 输入 | `[B, L, E]` | B 条文本，每条 L 个位置，每个位置 E 维 |
| `output` | `[B, L, D×H]` | 最后一层所有时间步的输出 |
| `h_n` | `[N×D, B, H]` | 每一层、每个方向的最终隐藏状态 |

其中 N 是层数，H 是隐藏维度，单向时 D=1，双向时 D=2。`batch_first` 不改变 `h_n` 的维度顺序；不设置它时，批量输入和 `output` 默认把序列长度放在第一维。

### 5.3 怎样做评论分类

一个基本模型是：`Token ID → Embedding → RNN → 句子表示 → 线性分类层`。

最后一层输出一个未归一化得分，称为 **Logit**。二分类可以对它做 Sigmoid，得到 0 到 1 之间的正面分数；训练时使用 `BCEWithLogitsLoss`，该函数内部已经包含数值稳定的 Sigmoid 与二分类交叉熵计算，前面不应再手动调用 Sigmoid。

这里有一个很常见的错误：不同长度的句子补齐后，直接取 `output[:, -1, :]`。短句的最后位置可能是 `<pad>`。即使 Embedding 的补齐向量为 0，循环网络仍可能更新隐藏状态。

配套的 [sequence_classifier.py](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/examples/sequence_classifier.py) 使用 `pack_padded_sequence()` 按真实长度处理输入，再取最后一层两个方向的最终状态。运行方式是：

```bash
python examples/sequence_classifier.py --arch rnn --out runs/rnn
```

这个脚本包含字符词表、`<unk>` 与 `<pad>`、自定义 `Dataset`、批次补齐、Adam 更新、验证集选模、独立测试、保存与重新加载。词表只从训练集构造；未知字符在验证和推理时映射到 `<unk>`。

### 5.4 RNN 为什么容易忘记前文

RNN 用 **BPTT（沿时间展开的反向传播）**训练。后面的损失要影响前面的参数，就需要沿多个时间步反向传递梯度。

可以先想象连续乘很多次 0.5：乘积很快接近 0；连续乘很多次 2，结果又会迅速变大。实际计算是矩阵及激活函数导数的连乘，但直觉类似。

前一种现象叫**梯度消失**，早期输入很难影响参数更新；后一种叫**梯度爆炸**，训练可能不稳定。梯度裁剪可以限制过大的梯度，却不能恢复已经消失的梯度。LSTM 和 GRU 就是在这样的背景下引入更有选择的信息传递机制。

<a id="s6"></a>

## 6 LSTM 和 GRU 如何选择记忆

### 6.1 LSTM 把保存和读取分开

**LSTM（Long Short-Term Memory，长短期记忆网络）**在隐藏状态 `h` 之外增加了细胞状态 `c`。可以先把 `c` 理解成沿时间传递的记忆，把 `h` 理解成当前对外提供的表示。

三个门共同控制信息流：

- **遗忘门**决定旧记忆保留多少。
- **输入门**决定新产生的候选信息写入多少。
- **输出门**决定当前从记忆中读出多少。

门不是手写的 `if`，而是模型根据当前输入和旧状态计算出的向量。向量里的每一项通过 Sigmoid 变成 0 到 1 之间的系数。

记忆更新的核心是：

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t
$$

$$
h_t=o_t\odot\tanh(c_t)
$$

`f_t`、`i_t`、`o_t` 分别是三个门，`g_t` 是候选记忆，`⊙` 表示逐元素相乘。

![LSTM 的记忆通路与 GRU 的状态混合，标出逐元素乘法和加法](11-gated-state-flow.png){: width="1600" height="1320" }

读上半张图时，先沿着旧记忆到新记忆的横线看：旧信息乘遗忘门系数，新信息乘输入门系数，再相加。输出门只控制怎样从新记忆得到隐藏状态，不会直接把细胞状态整条通路截断。

假如读到“这家店的包装很好，不过客服一直没有回复”，模型可以学习保留“没有回复”对最终评价的影响，而不只是机械地覆盖之前的状态。但每个门是否对应某种可解释的语言现象，需要观察，不能仅靠名字断定。

LSTM 的加性记忆路径使长期信息更容易传递。在只看这条直接路径时，梯度会受到遗忘门系数的影响；当需要保留的信息对应系数接近 1，衰减会更慢。**这不表示所有门都会自动接近 1，也不表示 LSTM 彻底解决了梯度问题。**

PyTorch 的 LSTM 返回 `output, (h_n, c_n)`。它也支持多层、双向及两者组合；没有投影时，`h_n` 和 `c_n` 的形状均为 `[N×D, B, H]`。设置 `proj_size>0` 后，隐藏状态使用投影维度，细胞状态仍使用 `hidden_size`，不能再把二者的形状当成相同。具体定义见 [PyTorch 2.6 循环模块接口与源码](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)。

### 6.2 GRU 用更少的门组织状态

**GRU（Gated Recurrent Unit，门控循环单元）**没有独立的细胞状态，主要使用两个门：

- 重置门控制在计算候选状态时参考多少过去的信息。
- 更新门控制旧状态与候选状态怎样混合。

按 PyTorch 的约定，可以把最终更新写成：

$$
h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}
$$

`z_t` 是更新门，`n_t` 是候选状态。有些资料对更新门的定义相反，公式中两个系数就会交换，阅读时需要看清约定。PyTorch 在候选状态中放置重置门的位置也与部分论文写法不同，详见 [PyTorch 2.6 GRU 接口与源码](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)。

相同隐藏维度下，GRU 通常比 LSTM 参数更少。但“更少”不等于在所有任务上更快、更准，最终还受实现、设备、数据和超参数影响。

### 6.3 用同一份数据比较才有意义

继续运行同一套分类程序：

```bash
python examples/sequence_classifier.py --arch lstm --out runs/lstm
python examples/sequence_classifier.py --arch gru --out runs/gru
```

比较时保持划分、词表、指标和计算预算一致，不要为某一个模型反复调参，却把另一个模型的默认结果拿来对比。

RNN、LSTM、GRU 的共同限制是：同一个序列的后续状态依赖之前的状态，所以时间步之间难以完全并行。门控改善了记忆方式，却没有取消这种依赖。接下来先看另一类输出问题，再理解注意力为什么有用。

<a id="s7"></a>

## 7 Seq2Seq 如何生成另一段话

### 7.1 分类输出固定标签  翻译输出可变序列

评论分类只要得到一个类别；把“我喜欢茶”翻译成英文，却要得到 `I like tea`。输入和输出长度未必相同，也不存在简单的一一对应关系。

**Seq2Seq（Sequence to Sequence，序列到序列）**把任务分成两部分：

1. **编码器 Encoder**读取源序列，得到表示。
2. **解码器 Decoder**根据源序列信息和已经知道的目标前文，逐步生成目标序列。

在经典、没有注意力的循环 Seq2Seq 中，编码器把源句压缩进一个定长状态，再用它初始化解码器。如果编码器是双向 GRU，常见做法是拼接两个方向的最终状态；如果解码器维度不同，还需要映射。

这里说的“压成一个向量”属于这类基础实现，**并不是所有 Seq2Seq 模型的共同限制**。后面的注意力模型会保留并读取源句的整个表示序列。

### 7.2 训练时为什么把答案交给解码器

先给目标句添加开始和结束标记：

```text
目标序列：<bos> I like tea <eos>
解码输入：<bos> I like tea
监督目标：      I like tea <eos>
```

`<bos>` 也常写作 `<sos>`；具体模型使用什么标记，要看其词表和配置。

训练时，在预测 `like` 的位置，把真实前文 `I` 提供给模型。这叫 **Teacher Forcing**。模型学习的是“给定正确前文，预测下一项”，而不是在每次训练时都先自己生成完整前文。

注意，输入必须与监督目标**错开一位**。如果把当前答案直接放到当前位置作为输入，任务可能变成抄答案。

每个目标位置输出一组词表得分，训练时用交叉熵衡量正确 Token 的得分是否足够高。补齐位置应通过 `CrossEntropyLoss(ignore_index=pad_id)` 等方式排除，并按有效 Token 汇总损失。`CrossEntropyLoss` 接收原始 Logits，不需要先做 Softmax。

### 7.3 推理时每一步使用自己的结果

真正翻译时没有参考译文。模型先接收 `<bos>`，生成 `I`；下一步把 `I` 加入前文，再生成 `like`，直到输出 `<eos>` 或达到最大生成长度。

这种把过去输出继续作为输入的过程称为**自回归生成**。它也带来训练与推理的差异：训练时看到正确前文，推理时可能看到自己的错误。前面的错误会影响后续输出，这常被称为暴露偏差。

模型给出的是候选词的得分，怎样选词还需要解码策略：

| 策略 | 做法 | 适合注意的取舍 |
| --- | --- | --- |
| 贪心解码 | 每一步选概率最高的 Token | 简单便宜，但局部最优不保证整句最优 |
| 束搜索 Beam Search | 保留多个候选前缀，按累计得分继续扩展 | 搜索范围更广、计算更多；仍不保证找到全局最优，也不必然提升质量 |
| 采样 | 按分布随机选择，可配合温度、top-k、top-p | 常用于增加生成多样性，但输出更不确定 |

束搜索通常累计对数概率，并可能引入长度惩罚，避免过度偏好短句。生成的“更可能”始终是相对于模型分布而言，不等于事实更准确。

### 7.4 跑通一个小型翻译模型

[translation.py](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/examples/translation.py) 内置少量人工编写的中英句对，不需要下载语料。中文按字符处理，英文按空格处理；两个语言分别建立词表。真实语料中，还需要更合适的分词和规范化方法，NLTK 是可选工具之一。

```bash
python examples/translation.py --arch gru --out runs/translation-gru
```

程序覆盖双向 GRU 编码器、单向解码器、目标右移、忽略补齐的损失、验证集选模、保存与重载、贪心生成，以及测试集上的 BLEU 计算。

**BLEU**主要比较译文与参考译文的 N-gram 重合，使用截断后的匹配计数，避免单纯重复一个词刷分，再结合长度惩罚。BLEU-4 汇总一至四元短语的匹配情况。

BLEU 不能充分衡量事实正确性、否定关系和多种合理表达；短句没有四元短语时，分数还对平滑方法特别敏感。配套示例采用 NLTK 的 `corpus_bleu` 和 `method1` 平滑，输出范围为 0 到 1。其他工具可能使用 0 到 100，比较前必须统一分词、参考译文、尺度与计算配置。具体接口见 [NLTK BLEU 文档](https://www.nltk.org/api/nltk.translate.bleu_score.html)。

<a id="s8"></a>

## 8 Attention 如何按需读取上下文

### 8.1 不必只依赖读完后的记忆

让编码器把很长的句子压到一个定长状态，会形成信息瓶颈。生成不同目标词时，需要参考的源位置也不同：生成商品名时关注商品，生成动作时关注“退款”或“发货”。

**Attention（注意力）**让解码器在每个生成步骤，重新为源句的各个位置分配权重，再读出一个适合当前步骤的上下文向量。

![普通 Seq2Seq 与带注意力的 Seq2Seq 对比](05-seq2seq-attention.png){: width="1600" height="820" }

它通常包含四步：比较当前解码状态与源位置的相关性；用 Softmax 得到权重；按权重汇总源表示；把汇总结果与解码状态融合，再预测目标词。

对于解码状态 `s` 和某个源位置的表示 `h`，常见打分方式有：

| 方式 | 示例公式 | 特点 |
| --- | --- | --- |
| Dot | `sᵀh` | 直接点积，要求对应维度一致 |
| General | `sᵀWh` | 加入可学习矩阵，既能对齐维度，也能学习匹配关系 |
| Concat | `vᵀ tanh(W[s;h]+b)` | 拼接后做非线性变换，再得到标量得分 |

点积大小不仅受方向影响，也受向量长度影响；非线性打分参数更多，也不代表一定更适合当前数据。

在同一个翻译脚本中，加入 Attention 的版本可以直接运行：

```bash
python examples/translation.py --arch attention --out runs/translation-attention
```

这个版本用解码器输出与全部编码器输出计算 Dot Attention，屏蔽源句补齐位置，再拼接上下文向量和解码器表示。RNN 仍然负责产生状态，因此它解决了按需读取的问题，但没有取消循环计算。

### 8.2 Query Key Value 怎么理解

Transformer 把这一过程整理成三个角色：

- **Query，查询向量**：当前位置想寻找怎样的信息。
- **Key，键向量**：每个位置用什么表示参与匹配。
- **Value，值向量**：匹配之后实际被汇总的信息。

这有点像查询与记录之间的匹配，不过它没有精确的字符串键比较。Q、K、V 都是模型学习出的数值投影，权重通常分布在多个位置上。

对于输入矩阵 X，**自注意力**从同一组位置得到三种投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

最常见的缩放点积注意力为：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V
$$

`d_k` 是 Key 的维度。缩放用于控制点积随维度增大时的幅度，避免 Softmax 过于尖锐；`M` 是掩码，允许的位置通常为 0，禁止的位置使用负无穷。Softmax 沿被读取的位置这一维计算。

公式里的上标 `T` 表示转置，也就是交换矩阵的行和列。`QK^T` 的第 i 行、第 j 列，就是第 i 个查询与第 j 个键的点积。

### 8.3 用一组小数字算一次

以下示例只依赖 NumPy，使用人为设定的向量：

```python
import numpy as np

X = np.array([[1., 0.], [0., 1.], [1., 1.]])
Q = K = V = X  # 为便于演算，令三个投影矩阵均为单位矩阵
scores = Q @ K.T / np.sqrt(2)
exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights = exp / exp.sum(axis=-1, keepdims=True)
output = weights @ V
print(weights.round(3))
print(output.round(3))
```

![数值注意力矩阵，每一行权重和为一](06-attention-math.png){: width="1600" height="840" }

第一行的权重约为 `[0.401, 0.198, 0.401]`，因此输出约为 `[0.802, 0.599]`。它不是挑中一个位置就结束，而是对多个 Value 做加权求和。完整的 [attention_numpy.py](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/examples/attention_numpy.py) 还演示了因果掩码。

NumPy 中的 `@` 是矩阵乘法，`.T` 是转置，`axis=-1` 表示沿最后一个维度计算。这里的每一行代表一个查询，最后一维列出了它可以读取的各个位置。

**注意力权重不是情感分类的概率，也不能直接当成模型决策的因果解释。** 这个算例只说明计算过程，没有经过语料训练，不能赋予它真实的语言含义。

<a id="s9"></a>

## 9 Transformer 如何组织这些计算

### 9.1 从逐步传递变成直接交换

RNN 中，句子开头的信息要经过许多状态才能到达末尾。自注意力允许某个位置直接读取其他位置的表示，缩短了信息交互的路径。

Transformer 将这一机制作为核心，配合前馈网络、位置编码、残差和归一化，构成可堆叠的层。原始 Transformer 是编码器—解码器结构，论文中的基础配置使用六层编码器和六层解码器。现代模型不必沿用这个层数。原始设计见 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

![Transformer 编码器、解码器和交叉注意力的关系](07-transformer.png){: width="1600" height="1000" }

### 9.2 多头注意力提供多个投影视角

单头注意力只使用一组 Q、K、V。**多头注意力**使用多组投影，分别计算注意力，把各头输出拼接后再做一次线性映射。

可以理解成允许模型同时从多个表示子空间读取信息。有些头可能形成局部、指代或其他关系模式，但这不是预先规定的分工；不能说“第一头专门学语法，第二头专门学语义”。

在常见实现中，`d_model=512`、`nhead=8` 表示每个头的维度为 64，模型总维度仍为 512，不是把八份 512 维表示简单堆上去。PyTorch 的标准实现要求 `d_model` 能被 `nhead` 整除。

![同一输入经两组不同投影得到两个注意力头，拼接后再映射](12-multihead-attention.png){: width="1600" height="900" }

图中用 8 维、2 个头简化演示。每个头都从完整的输入表示学习自己的 Q、K、V 投影，并不是先把原句分成两半、每个头只看一半。最后的输出投影会进一步混合各头的信息。

### 9.3 FFN 残差和 LayerNorm 各做什么

注意力负责不同位置之间的信息交换，**FFN（前馈网络）**负责对每个位置的表示做非线性变换。标准形式是两层线性变换，中间接 ReLU 等激活函数，先扩展维度，再投影回来。同一层的各位置共用 FFN 参数。

**残差连接**把子层输入直接加回输出，例如 `x + Attention(x)`。模型可以在已有表示上学习修正，也给梯度提供额外的传播路径，但并不保证任意深度的训练都稳定。

**LayerNorm（层归一化）**通常对每个 Token 的特征维计算均值和方差，再进行标准化以及可学习的缩放、平移。它不像 BatchNorm 那样主要依赖一批样本的统计量。经过可学习的缩放和平移后，最终结果也不必严格保持均值 0、方差 1。

原始 Transformer 常用“残差之后归一化”的 Post-LN；许多后续模型使用“子层之前归一化”的 Pre-LN。读代码时看到 `norm_first=True`，就是在改变这项安排。Dropout 则在训练时随机丢弃部分激活，用于正则化，推理时通常关闭。

### 9.4 为什么还需要位置编码

没有位置信息时，标准的全可见自注意力对位置重排具有相应的等变性：Token 跟着重排，输出表示也跟着重排。它无法仅凭这些运算识别完整语序。因此，我们需要告诉模型一个 Token 出现在哪里。

原始 Transformer 使用正弦、余弦位置编码，把不同频率的波形组合成位置向量，再加到 Token 向量上。它不需要训练，每个位置都有固定编码。简单使用越来越大的位置整数会改变输入数值尺度；按句长归一化又会让同一位置随句长变化。

![32 维正余弦位置编码的不同频率曲线与位置乘维度热力图](13-position-encoding.png){: width="1600" height="950" }

图左只画出三组维度：有的变化快，有的变化慢；右侧每一行则是一个位置的完整 32 维编码。把这些数加到词向量上，就让相同 Token 出现在不同位置时具有不同的输入表示。它提供位置线索，不会直接告诉模型哪种语序才正确。

位置编码也可以是可学习的嵌入。后续模型还采用相对位置方法、RoPE 等设计。因此，“Transformer 一律使用正余弦位置编码”并不成立；不同模型有不同的位置机制。原始正余弦公式放在附录中。

### 9.5 编码器和解码器的注意力有什么不同

编码器的自注意力通常能读取源序列的所有有效位置；生成式解码器的自注意力需要遵守因果限制，只能读取自身及之前的输入位置。

编码器—解码器结构里，还有**交叉注意力**：Query 来自解码器，Key 和 Value 来自编码器输出。自注意力回答“这段序列内部怎样相互关联”，交叉注意力回答“当前目标位置应该读取源序列的哪些信息”。

这里有两种常被混淆的掩码：

| 掩码 | 屏蔽什么 | 常见形状 |
| --- | --- | --- |
| Padding Mask | 补齐出来的无效位置 | `[B, L]` |
| Causal Mask | 当前查询位置之后的未来输入位置 | `[L, L]` |

这里的 Padding Mask 通常屏蔽的是作为 Key 被读取的补齐列，**不保证补齐位置作为 Query 得到的输出向量为零**。因此，计算逐 Token 损失时仍要排除 PAD；对句子做平均池化时，也要按有效位置筛选。

![目标右移、因果掩码与自回归推理的区别](08-train-generate-mask.png){: width="1600" height="890" }

输入右移后，当前位置的输入是已知前文中的最后一个 Token，所以因果掩码可以保留对角线。例如输入位置是 `I` 时，该位置预测的是 `like`，读取自身的 `I` 并不泄漏答案。

**不同 API 的布尔掩码含义可能相反。** `nn.Transformer` 中 `True` 通常表示禁止读取，而 `torch.nn.functional.scaled_dot_product_attention` 的布尔 `attn_mask` 中 `True` 表示允许读取。Hugging Face 常见的 `attention_mask` 又通常以 1 表示有效位置、0 表示补齐。应按各自接口转换，不能直接复用。这个差异在 [PyTorch Transformer 文档](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Transformer.html)中有明确说明。

### 9.6 为什么训练能并行  生成还要一步步来

训练时，完整目标序列已经存在。把目标右移并加上因果掩码，就可以同时计算多个位置的下一词预测，而不用像 RNN 那样按时间步更新状态。

推理时，下一个 Token 还不知道，常规自回归解码仍需逐步生成。但是，**不必每一步都把整个前缀的所有计算从头再做一遍**。实际系统常使用 KV cache 保存之前的 Key、Value，以减少重复计算。缓存通常按层、按请求保存，并会随前缀长度增长，占用更多内存。

本文的 `nn.Transformer` 教学示例没有实现 KV cache，生成时会重新计算已有前缀；这是便于观察计算过程的简化。生产推理中能否使用缓存，要看具体模型和生成接口，不能以为接入 `nn.Transformer` 就自动获得了它。

并行训练也不意味着注意力没有成本。标准稠密注意力的分数矩阵有 `L×L` 个元素，相关计算量通常随序列长度平方增长。高效注意力实现可以减少中间矩阵的显存读写和存储开销，但不能简单理解为长上下文从此免费。

### 9.7 PyTorch 帮我们实现了哪一部分

`nn.Transformer` 提供编码器—解码器核心，`TransformerEncoder` 和 `TransformerDecoder` 分别封装对应的多层结构；`TransformerEncoderLayer`、`TransformerDecoderLayer` 表示单层。

它们接收的通常是浮点表示，不是原始字符串，也不是直接可用的 Token ID。你还需要添加源端与目标端 Embedding、位置编码，以及把隐藏维映射到词表大小的输出层。

完整示例已经包含这些部分：

```bash
python examples/translation.py --arch transformer --out runs/translation-transformer
```

它调用 `transformer.encoder` 得到源端 `memory`，再通过 `transformer.decoder` 读取目标前缀；源补齐掩码同时用于编码器自注意力和解码器交叉注意力。

在这份极小语料上，Transformer 并不保证胜过 GRU。运行这个例子的主要收获，是亲自确认：输入如何右移、掩码如何传递、损失如何计算，以及预测序列如何结束。

<a id="s10"></a>

## 10 预训练如何复用语言知识

### 10.1 每个业务都从零训练太昂贵

如果每次做评论分类，都从随机参数开始让模型学习语言，既需要大量数据，也会重复学习词语和上下文规律。

**预训练**先在大规模文本上学习可复用的表示。很多训练目标直接从原始文本构造标签，例如“预测下一个词”或“恢复被遮盖的片段”，因此更准确地说属于**自监督学习**。

**微调（Fine-tuning）**再用业务数据调整预训练模型。调整对象可以是全部参数，也可以只是任务头或部分新增参数。它通常比完全从零训练更节省数据，但不是没有标注、没有评估也能自动成功。

![GPT、BERT、T5 的结构和预训练目标对比](09-pretraining-objectives.png){: width="1600" height="840" }

### 10.2 GPT 学习根据前文预测下一项

GPT 是生成式预训练模型的一条代表路线。经典 GPT 使用 Decoder-only 结构，核心是带因果掩码的自注意力和 FFN，**没有原始编码器—解码器结构中的那层交叉注意力**。

它把一个序列的概率拆成连续的条件概率：预测当前 Token 时，只依赖前面的 Token。语言模型头把每个位置的隐藏表示映射到词表得分，预训练时以真实的下一项作为监督目标。

GPT-1 在 BooksCorpus 等具体实验设置下展示了生成式预训练与下游任务迁移的价值。其经典配置是 12 层、12 个注意力头、768 维隐藏表示，使用可学习的位置嵌入；这些数字和位置方案不能推广到整个 GPT 家族。原始方法与实验背景见 [GPT-1 论文](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)。

GPT 也可以做分类。例如，将单句、句子对或问题与候选答案转换成统一的输入序列，取指定位置的表示，再接一个分类输出层。GPT-1 的一种分类形式可以示意为“起始标记 → 评论内容 → 分类用结束标记”，再读取末尾位置的隐藏表示进行分类。末尾已经能看到前面的整条评论；具体标记由模型定义，不要把示意名称硬编码给其他 GPT 模型。现代对话模型还涉及指令训练等后续步骤，“会续写”与“能按业务要求回答”不能直接画等号。

### 10.3 BERT 同时利用左右上下文

BERT 使用 Encoder-only 结构。对于一个已经完整给出的句子，它可以同时读取左右两侧的内容，适合分类、匹配和抽取等任务。

经典 BERT 的输入由三部分相加：Token Embedding、可学习的位置 Embedding，以及区分句子 A/B 的 Segment Embedding。常见输入形式是：

```text
单句：[CLS] 评论内容 [SEP]
句对：[CLS] 第一句 [SEP] 第二句 [SEP]
```

`[CLS]` 是常用于分类的特殊位置，`[SEP]` 用于分隔句子或标记边界。`token_type_ids` 指定句子片段类型，通常以 0、1 区分句子对；它与区分补齐位置的 `attention_mask` 是两件事。

原始 BERT 使用两个预训练目标：

**MLM（Masked Language Modeling，掩码语言模型）**随机选择约 15% 的 Token 作为预测目标。在这些被选中的位置里，80% 换成 `[MASK]`，10% 换成随机 Token，10% 保持原样。不是“所有输入 Token 都有 80% 被遮住”。模型需要利用可见上下文预测原 Token。

**NSP（Next Sentence Prediction，下一句预测）**判断句子 B 是否是句子 A 的真实后续。原始设置中，一半样本是真实连续句，一半是随机配对。它是二分类任务，不是生成下一句话。后续模型可能修改或去掉这个目标。经典结构与训练细节见 [BERT 论文](https://aclanthology.org/N19-1423/)。

微调时，输出头按任务变化：

| 任务 | 使用什么表示 | 输出 |
| --- | --- | --- |
| 单句分类 | 常用 `[CLS]` 位置 | 情感、主题等类别 |
| 句子对分类或回归 | 将句子对一起编码，常用 `[CLS]` | 是否重复、是否蕴含或相似度得分 |
| 序列标注 | 每个有效 Token 的表示 | BIO 实体标签等 |
| 抽取式问答 | 问题和段落的 Token 表示 | 答案起点、终点的得分 |

BIO 中，B 表示实体开始，I 表示实体内部，O 表示不属于实体。例如 `B-PER` 是人名开始。子词分词后，原始词级标签与子词位置还需要对齐，不能直接沿用原始索引。

问答输出起止位置，也不代表任何输入都一定含有正确答案；是否支持“无答案”，取决于任务定义和训练数据。

### 10.4 T5 把任务统一为文本到文本

T5 使用 Encoder–Decoder 结构，并把多种任务统一成文本输入与文本输出。例如，情感分类也可以输出字符串 `positive`，翻译任务可以通过任务前缀说明要做什么。

其代表性预训练目标是**片段破坏与恢复（Span Corruption）**。把连续片段换成不同的哨兵标记，再让解码器生成缺失片段。一个按词简化的例子是：

```text
原文：订单 已经 退款，请 留意 到账
输入：订单 <extra_id_0>，请 <extra_id_1> 到账
目标：<extra_id_0> 已经 退款 <extra_id_1> 留意 <extra_id_2>
```

哨兵标记帮助模型区分多个缺失片段。T5 不必生成整段未损坏的原文。微调时，则按具体任务构造输入文本与目标文本。

| 任务 | 输入格式示例 | 目标文本 |
| --- | --- | --- |
| 翻译 | `translate English to German: That is good.` | `Das ist gut.` |
| 情感分类 | `sentiment: The delivery was fast.` | `positive` |
| 问答 | `question: What is the order status? context: The order was refunded.` | `refunded` |

这些例子说明如何组织训练样本；前缀、语言和标签必须与具体检查点的训练设置匹配，单纯添加一个任务前缀并不会创造任务能力。

这条路线的价值是统一接口和训练表达形式，但并不意味着一个任意的 T5 检查点天然支持所有语言和任务。原始研究及代码入口见 [T5 论文](https://jmlr.org/papers/v21/20-074.html)；其预印本发表于 2019 年，期刊版本发表于 2020 年。

### 10.5 实际选择先看任务和数据

分类、实体抽取、句子匹配可以从编码器模型开始；翻译、摘要可以考虑已经在相应任务上训练过的编码器—解码器模型；开放式续写和对话经常使用 Decoder-only 模型。这是选型起点，不是能力边界。

对业务开发更直接的问题是：模型支持哪种语言，是否训练过当前任务，输入能多长，延迟与内存是否合适，以及真实测试集上的表现如何。只比较参数数量，回答不了这些问题。

<a id="s11"></a>

## 11 认识 Hugging Face 的分工

Hugging Face 既提供模型与数据的托管平台，也提供开发库。把它理解成“一个大模型”会混淆几个层次。

![Hub、Datasets、Tokenizer、Transformers 与 PyTorch 的协作流程](10-hf-workflow.png){: width="1600" height="900" }

| 组件 | 主要工作 | 后端工程中的近似理解 |
| --- | --- | --- |
| Hub | 托管、查找和版本化模型、数据集与应用 | 带模型说明的制品仓库 |
| Transformers | 加载模型结构、权重与任务头，提供推理及训练接口 | 统一模型开发接口 |
| Tokenizers | 高效实现分词算法和编码组件 | 文本编码的底层工具 |
| `AutoTokenizer` | 在 Transformers 中加载匹配模型的分词器 | 按模型配置选择编码适配器 |
| Datasets | 加载、变换、缓存和保存数据集 | 面向模型数据的处理工具 |
| PyTorch | 张量计算、自动求导、神经网络和优化器 | 负责实际计算与训练的框架 |
| Spaces | 托管模型应用或演示 | 可交互的应用展示环境 |

这些类比帮助理解职责，但不是完全相同的系统。例如，模型仓库里的说明卡还会描述训练用途、标签、局限和许可，不能只下载权重而忽略这些信息。

Hugging Face 生态也覆盖图像、音频和多模态，Datasets 并不限于文本。`pipeline()` 适合快速调用已有任务能力；希望看清中间过程时，可以直接使用 Tokenizer 和模型。整体接口见 [Transformers 快速入门](https://huggingface.co/docs/transformers/en/quicktour)。

<a id="s12"></a>

## 12 加载模型并读懂输入输出

### 12.1 `from_pretrained` 实际做了什么

`AutoModel.from_pretrained("google-bert/bert-base-chinese")` 通常会解析仓库配置，确定模型结构，下载或读取权重，再把权重加载到相应的模型实例。

其中 `config.json` 描述结构，权重文件保存学习到的数值。只构造一个配置并创建网络，不等于已经加载预训练知识。`AutoModel` 主要提供主干的隐藏表示；任务模型则额外包含适合某个任务的输出头。

| 类 | 常见用途或输出 |
| --- | --- |
| `AutoModel` | 每个位置的上下文隐藏表示 |
| `AutoModelForSequenceClassification` | 整段文本的分类或回归 |
| `AutoModelForTokenClassification` | 每个 Token 的标签 |
| `AutoModelForQuestionAnswering` | 抽取式问答的起止位置 |
| `AutoModelForMaskedLM` | 被遮盖位置的词表预测 |
| `AutoModelForCausalLM` | 基于前文的自回归语言建模 |
| `AutoModelForSeq2SeqLM` | 编码器—解码器文本生成 |

选类时要同时看任务与架构兼容性。例如，不能通过换一个类名，就让任意 BERT 权重变成已经训练好的翻译模型。

`num_labels` 控制分类输出维度。**从基础 BERT 加载一个分类类，分类头可能是新初始化的。** 警告中的 `newly initialized` 往往就是在提醒你：这部分权重还没有针对任务训练。不能把此时得到的正负分数当成可用的情感分析结果。

### 12.2 一个完整的 BERT 前向示例

保存并运行下面的代码即可。它读取两条中文文本，输出每个位置的向量，不作情感判断。

```python
import torch
from transformers import AutoTokenizer, AutoModel

name = "google-bert/bert-base-chinese"
revision = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"
tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
model = AutoModel.from_pretrained(name, revision=revision, use_safetensors=True)
inputs = tokenizer(
    ["物流很快，包装完整", "有破损"],
    padding=True, truncation=True, max_length=32, return_tensors="pt"
)
model.eval()
with torch.inference_mode():
    outputs = model(**inputs)
print({key: tuple(value.shape) for key, value in inputs.items()})
print(tuple(outputs.last_hidden_state.shape))
print(tuple(outputs.pooler_output.shape))
```

在这组输入和固定模型版本上，实际输出的形状为：

```text
input_ids       (2, 11)
token_type_ids  (2, 11)
attention_mask  (2, 11)
last_hidden_state  (2, 11, 768)
pooler_output      (2, 768)
```

第一维是批量大小 2，第二维 11 包括特殊标记与补齐，768 是这个模型的隐藏维度。`last_hidden_state` 保留每个位置的表示；这个 BERT 的 `pooler_output` 是对 `[CLS]` 隐藏状态进行线性变换、再经过 `tanh` 的结果，**不是简单平均所有位置**，也不保证能直接作为高质量语义检索向量。字段与形状以 [BERT 的模型接口](https://huggingface.co/docs/transformers/en/model_doc/bert)为准。

![BERT 句子对中的 Token、ID、片段编号、位置编号和补齐掩码逐列对齐](14-bert-input-fields.png){: width="1600" height="1060" }

图中第一句是“我爱茶”，第二句是“很好”。`token_type_ids` 从第二句开始变为 1，`attention_mask` 在末尾 PAD 处为 0。图中的默认位置编号通常由 BERT 内部构造，并非 Tokenizer 必然返回的字段。输入表示由词、位置和片段三种 Embedding 相加，掩码则另外传入注意力计算；它不是第四种 Embedding。

`model.eval()` 切换 Dropout 等模块的行为，`torch.inference_mode()` 关闭推理中不需要的梯度跟踪。二者解决不同问题，不能相互替代。实际调用时使用 `model(**inputs)`；查看某个模型的 `forward()` 定义，可以确认它支持哪些字段。不同架构未必都有 `token_type_ids` 或 `pooler_output`。

如果把模型移到 GPU，输入张量也要移到相同设备。CPU 示例刻意省略设备选择，以便先专注于输入输出契约。

### 12.3 Tokenizer 的几个接口分别做什么

| 方法 | 作用 | 容易误解的地方 |
| --- | --- | --- |
| `tokenize(text)` | 得到 Token 字符串列表 | 通常不会完成模型输入的全部包装 |
| `convert_tokens_to_ids(tokens)` | 按词表查 ID | 单纯查表，不代表添加了特殊符号 |
| `convert_ids_to_tokens(ids)` | 按 ID 查回 Token | Token 列表还不是自然文本 |
| `encode(text)` | 编码为 ID 序列，通常加入特殊标记 | 可用 `add_special_tokens=False` 控制 |
| `decode(ids)` | 将 ID 解码为文本 | 可跳过特殊标记，但未必逐字恢复原始空格和格式 |
| `tokenizer(texts, ...)` | 构造模型输入字典，支持批量 | 通常是训练和推理中最方便的入口 |

对上面这个中文 BERT，`tokenize("我爱自然语言处理")` 得到八个汉字 Token；`encode()` 还会加上 `[CLS]` 和 `[SEP]`。`decode(..., skip_special_tokens=True)` 的结果可能带有字间空格，这不表示模型把原文保存成了逐字完全可逆的压缩格式。

补齐和截断尤其需要看清参数：

- `padding=True`：补到当前 batch 最长序列，不是自动补到 `max_length`。
- `padding="max_length"`：补到指定最大长度。
- `truncation=True`：对超过限制的输入截断。
- `max_length`：限制的是编码后的 Token 数量，通常包含特殊标记，不是汉字数或字节数。
- `return_tensors="pt"`：返回 PyTorch 张量。

为了少做无效计算，训练中经常采用动态补齐，即只把同一个 batch 补到所需长度。更详细的参数组合见 [Padding 与 Truncation 文档](https://huggingface.co/docs/transformers/main/pad_truncation)。

输入过长时，截断有可能刚好切掉最后的“不推荐”。应结合任务选择切片、分段处理或其他长文本策略，而不是只确保代码没有报错。

### 12.4 模型缓存和真正的离线运行

Hub 默认缓存通常位于 `~/.cache/huggingface/hub`，可以用 `HF_HOME`、`HF_HUB_CACHE` 或相应方法的 `cache_dir` 调整。缓存避免重复下载未变化的文件，但在线模式仍可能请求元数据确认版本。目录结构和行为见 [Hub 缓存文档](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)。

复现实验时，除了 Python 包版本，还要固定模型的 `revision`，最好使用具体提交标识。上面示例已经固定了一个版本，模型页面见 [bert-base-chinese 模型卡](https://huggingface.co/google-bert/bert-base-chinese)。

模型与分词器应一起保存：`model.save_pretrained(path)`、`tokenizer.save_pretrained(path)`。之后把本地目录传给 `from_pretrained()`，并设置 `local_files_only=True`；或在导入相关库前设置 `HF_HUB_OFFLINE=1`。离线运行的前提是所需文件已经完整存在。相关环境变量见 [Hub 环境变量文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)。

完整脚本不仅做前向计算，还会保存、重新加载并检查隐藏表示一致：

```bash
python examples/hf_inspect.py --out runs/bert-local
```

有些网络环境通过 `HF_ENDPOINT` 指向镜像或内部代理。它改变的是请求目标，不会替你解决模型版本、访问权限和制品一致性问题，也不应把第三方镜像当作官方服务。生产环境更适合管理已经核验的模型制品及其版本。

<a id="s13"></a>

## 13 用 Datasets 准备训练数据

### 13.1 先分清 Dataset 和 DatasetDict

`Dataset` 可以理解成带类型信息的数据表；`DatasetDict` 则是按名称组织多个 Dataset 的字典，例如 `train`、`validation`、`test`。

`load_dataset("csv", data_files="reviews.csv")` 默认返回一个包含 `train` 的 DatasetDict。**这个默认名字并不表示库替你做好了科学的数据划分。** 使用 `split="train"` 可以直接取得对应 Dataset。加载多个文件时，`data_files` 可以传入把 split 名称映射到文件路径的字典。

除了 CSV，还可以加载 JSON、Parquet，或用 Hub 数据集标识加载在线数据。在线加载时还可能需要选择配置与 split，并查看数据集卡了解字段、语言、许可和既有划分。接口说明见 [Datasets 加载文档](https://huggingface.co/docs/datasets/en/loading)。

查看数据时，`dataset[0]` 是一条样本字典；`dataset[:3]` 通常是“列名 → 多个值”的字典，而不是三个样本字典构成的列表。这个差别在批处理时很重要。使用 `dataset["text"]` 可以取得某一列；在本文锁定的 Datasets 4.2 中，它是可索引、可迭代的 `Column` 对象，不应假定在所有版本里都是普通 Python 列表。确实需要完整列表时，可以使用 `list(dataset["text"])`。

### 13.2 预处理是一条可以复现的管线

可以按以下次序处理评论数据：

1. 确认 `text`、`label` 的业务含义，过滤空文本和非法标签。
2. 检查重复、近重复以及同一订单或用户的关联样本，确定划分方式。
3. 保留独立测试集，再从开发数据中划出验证集。
4. 用固定的 Tokenizer 编码，限制长度，保留标签。
5. 在组成 batch 时补齐，并构造模型需要的字段。

常见 API 包括 `remove_columns()` 删除列、`filter()` 过滤行、`train_test_split()` 划分数据、`map()` 编码。需要分层抽样时，`stratify_by_column` 对标签列的类型有要求；整数标签可以先用 `class_encode_column("label")` 转成 ClassLabel。默认返回的 `test` 键可以在你的流程里作为验证集，但应明确命名，避免后来误当成最终测试集。

**`map(..., batched=True)` 传入的是一批数据组成的列字典。** 例如 `batch["text"]` 是一组文本，而不是 `batch` 本身就是样本列表。

下面的完整示例读取配套 CSV 的既有训练划分，进行编码和动态补齐：

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

raw = load_dataset("csv", data_files="examples/reviews.csv")["train"]
train = raw.filter(lambda row: row["split"] == "train")
train = train.remove_columns("split")
tokenizer = AutoTokenizer.from_pretrained(
    "google-bert/bert-base-chinese",
    revision="8f23c25b06e129b6c986331a13d8d025a92cf0ea"
)

def encode(batch):
    return tokenizer(batch["text"], truncation=True, max_length=64)

encoded = train.map(encode, batched=True, remove_columns=["text"])
encoded = encoded.rename_column("label", "labels")
loader = DataLoader(
    encoded, batch_size=4, shuffle=True,
    collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
)
batch = next(iter(loader))
print({key: tuple(value.shape) for key, value in batch.items()})
```

Data Collator 是把若干样本整理成一个 batch 的函数或对象。这里由它把长短不一的 ID 列表补齐，再转成张量。`labels` 是任务模型常用的监督字段名。

也可以提前把所有序列补到同一长度，再使用 `set_format(type="torch", columns=[...])` 配合默认 DataLoader。`set_format()` 改变取样时的返回视图，不会改写底层数据存储；`with_format()` 则返回带新格式的对象。二者都不会自动解决变长序列的补齐问题。数据处理方式见 [Datasets Process 文档](https://huggingface.co/docs/datasets/en/process)。

### 13.3 保存和缓存也有不同层次

`save_to_disk()` 可以保存 Dataset 或 DatasetDict，`load_from_disk()` 恢复其结构和元数据。数据通常以 Arrow 等库内部格式保存，大数据可能分成多个分片，并非永远每个 split 只有一个文件。

需要给其他工具读取时，可以对单个 split 使用 `to_csv()`、`to_json()`，再用 `load_dataset()` 载入。默认 JSON 导出常是按行的记录；CSV、JSON 重新加载时可能重新推断类型，不应假设 ClassLabel 等元数据完整保留。

Hub 下载缓存和 Datasets 处理缓存也需要区分：下载制品通常使用 Hub 缓存；Datasets 的处理结果默认常位于 `~/.cache/huggingface/datasets`，可通过 `HF_DATASETS_CACHE` 等方式调整。缓存目录不是业务数据划分的替代物，也不保证离线缺失文件能被自动补全。

完整示例包含过滤、删列、分层抽样、批量编码、Arrow 保存恢复、CSV/JSON 导出及 DataLoader：

```bash
python examples/datasets_demo.py --model runs/bert-local --out runs/datasets
```

这条命令复用上一节保存的 Tokenizer。具体缓存配置见 [Datasets 缓存说明](https://huggingface.co/docs/datasets/en/cache)。

<a id="s14"></a>

## 14 完成一次评论分类微调

### 14.1 把数据和模型接口接起来

现在，数据端可以提供 `input_ids`、`attention_mask`、可选的 `token_type_ids` 和 `labels`；模型端可以通过 `AutoModelForSequenceClassification` 接收这些字段。

我们使用两个输出类别：0 表示负面，1 表示正面。分类头输出形状为 `[B, 2]` 的 Logits，模型根据整数标签计算交叉熵。`id2label` 与 `label2id` 把模型编号和业务标签对应起来，避免推理时只看到意义不明的 `LABEL_0`。

注意，这与前面循环网络的“一个 Logit + 二分类交叉熵”是两种合法的二分类表达，不能把损失和标签格式随意混用。多标签任务允许一条文本同时属于多个类别，通常需要独立的 Sigmoid 输出及不同标签形式；它也不同于二选一或多选一的分类。

运行随文提供的完整微调程序：

```bash
python examples/hf_finetune.py \
  --model runs/bert-local \
  --epochs 1 \
  --out runs/bert-review
```

这里默认使用全部模型参数进行微调。示例程序会加载已有 BERT 主干，初始化分类头，用 AdamW 更新参数，裁剪过大的梯度，用验证损失选择保存的模型，再恢复该模型完成测试和预测。代码见 [hf_finetune.py](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/examples/hf_finetune.py)。

如果想观察冻结主干、只训练分类头的差别，可以增加 `--freeze-backbone`。冻结通过 `requires_grad=False` 控制；是否启用 Dropout 又由 `train()`、`eval()` 控制，两者不是同一个开关。为了让冻结特征在这个教学例子中保持稳定，脚本还把冻结的主干切换到 `eval()`。

### 14.2 一次训练迭代在做什么

训练循环的顺序始终可以拆成五步：

1. `optimizer.zero_grad()` 清除上一次累积的梯度。
2. `model(**batch)` 做前向计算，取得包含损失的输出。
3. `loss.backward()` 计算梯度。
4. 必要时进行梯度裁剪。
5. `optimizer.step()` 更新参数。

验证时切换 `model.eval()`，并关闭梯度跟踪；验证结束、继续训练时，再调用 `model.train()`。记录训练损失和验证损失，有助于发现过拟合：如果训练损失继续降低，验证表现却恶化，记住训练样本并没有转化成更好的泛化。

TensorBoard 的 `SummaryWriter.add_scalar()` 可以记录这些曲线。如果之后改用 Transformers 的 `Trainer`，它只是帮你管理训练和评估循环，不会替你定义正确的业务标签或数据划分。

### 14.3 保存之后还要确认能重新推理

微调模型保存在 `runs/bert-review`，目录包含权重、配置、标签映射和 Tokenizer。完成上面的微调并生成该目录后，再单独运行下面的例子：

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="runs/bert-review",
    tokenizer="runs/bert-review",
    device=-1
)
print(classifier("包装破损，客服没有回复", truncation=True, max_length=64))
```

这里 `device=-1` 表示 CPU。返回的 `score` 是这个模型给出的类别分数，不代表经过校准的“真实正确率”。

**这份教学数据只有 24 条训练样本、8 条验证样本和 8 条测试样本。** 在本文的一次 CPU 验证中，BERT 微调一轮后测试准确率为 0.50，甚至把上面的负面评价判成了正面；同一份数据上的 TF-IDF 基线准确率为 0.75。结果说明训练、保存和推理链路已经跑通，也说明当前模型还远不能直接投入业务。

这几个数字不用于比较模型架构优劣：样本太少，领域覆盖不足，超参数也没有经过充分选择。把失败样本保留下来，比只挑一条预测正确的句子展示更有学习价值。

<a id="s15"></a>

## 15 从教学示例走向业务服务

### 15.1 先定义模型到底帮业务做什么

假设评论分类用于“把疑似负面评价送进人工队列”，业务目标未必是整体准确率最高。漏掉严重投诉的成本，可能高于多送几条普通评价给人工。

因此，除了准确率，还应看：

- **Precision，精确率**：送入目标类别的样本中，有多少判断正确。
- **Recall，召回率**：实际属于目标类别的样本中，找出了多少。
- **F1**：精确率与召回率的调和平均；Macro-F1 对各类别分别计算后取平均，便于观察小类别。
- 按商品、渠道、文本长度和时间切片的表现，以及错误样本本身。

这里的“正类”是统计定义，可以设为“需要关注的负面评价”，不必等于情感标签里的“正面”。讨论指标前，应先统一哪个类别是目标。

阈值也不是只能设为 0.5。可以在验证集上根据业务代价选择，但不能反复使用测试集调阈值，再把同一个测试集成绩当作独立结论。生成任务除了 BLEU 等自动指标，还需要检查事实、遗漏、术语和人工可接受性。

### 15.2 数据问题往往比模型结构更先决定结果

随机划分不一定符合真实业务。相同订单的多条评价、几乎一样的模板文本、同一用户的关联记录，如果分别出现在训练与测试中，会让分数显得过于乐观。按订单、用户或时间划分，有时更接近真实上线条件。

自建词表、TF-IDF 的统计、特征选择等都应只在训练数据上拟合。预训练 Tokenizer 已有固定词表，但业务清洗、长度选择和阈值等仍然可能受到测试数据影响。

清洗也不能只是“删除看起来不重要的词”。“不”“没”“但是”会改变情感关系，表情、标点和拼写变化也可能包含信号。应先检查任务，明确每一种清洗规则会损失什么信息。

### 15.3 后端接入时仍然需要熟悉的工程能力

一个可用的推理服务，应在启动时加载模型和 Tokenizer，避免每个请求重复下载、初始化。请求进入后，检查输入格式和长度，组成合适的 batch，执行推理，再把标签或生成结果映射回业务响应。

模型制品需要版本号，记录权重、Tokenizer、依赖、标签映射、最大输入长度和评估结果。服务监控除了错误率、延迟和吞吐，还应观察输入长度分布、截断比例、类别分布，以及人工反馈的漂移。

长文本、空文本、超时和资源不足都需要明确处理方式。分类结果可以进入人工队列；涉及订单状态的回答，应由可靠的业务数据提供事实依据，而不是让语言模型自行猜测。

把模型装进 HTTP 接口，只解决了调用方式。模型是否适合当前业务，需要用离线评估、灰度流量和实际反馈继续确认。

### 15.4 常见误区集中检查

| 误区 | 更准确的理解 |
| --- | --- |
| Token 就是一个完整词 | 它也可能是字符、子词或字节单位 |
| Embedding 的每一维都有明确含义 | 多数维度是联合学习出的数值特征，通常没有预设名称 |
| 模型能加载就能做任何任务 | 架构、任务头和训练目标都必须匹配 |
| 新建分类头后马上就能做情感分析 | 新头需要训练，标签映射也需要确认 |
| BERT 的 `[CLS]` 或平均向量天然适合检索 | 检索表示通常还需要适合的训练目标、池化方式和评估 |
| 注意力图就是模型为什么判断的完整解释 | 它只展示某层某次信息汇总的权重 |
| LSTM 彻底解决长依赖问题 | 门控只是缓解，长度、梯度和顺序计算的限制仍在 |
| Transformer 可以一次并行生成所有未知词 | 常规自回归生成仍有逐步依赖，缓存用于减少重复计算 |
| 分数 0.95 就表示有 95% 把握正确 | 模型分数可能未校准，领域变化还会影响可靠性 |
| 下载有缓存就已经可以离线运行 | 可能仍有元数据请求，离线前要确保制品完整并配置离线模式 |
| 训练损失很低就可以上线 | 还要看独立评估、业务代价、延迟、资源和失败处理 |

### 15.5 下一步怎样学

第一轮先跑通 TF-IDF 基线和 BERT 输入输出，观察文本经过 Tokenizer 后究竟是什么。尝试更换一句话、加入一条很长的输入，再查看形状和截断行为。

第二轮阅读注意力的 NumPy 例子，亲手修改 Q、K、V，确认每行权重和为 1，以及未来位置如何被遮挡。然后切换翻译脚本的三个架构，把公式与实际张量对应起来。

第三轮换成经过确认的真实业务数据。先完善标签定义、划分和错误分析，再考虑增加训练轮数或调整模型。达到这一步，学习参数高效微调（例如 LoRA）才会有清楚的问题背景。

如果目标是知识问答，可以继续学习向量检索与 RAG：前者寻找相关材料，后者把检索到的材料作为生成依据。它们分别涉及表示、检索、生成和证据验证，不能只凭“接入一个大模型”就认定整个系统已经可靠。

下面的附录保留了公式和 API 的查阅入口。初读时不必全部记住；在写代码遇到形状或掩码问题时，再回来对照更有效。

<a id="appendix"></a>

## 附录 公式 API 与术语速查

### A 几个需要看懂的公式

余弦相似度比较两个非零向量的方向：

$$
\cos(u,v)=\frac{u^Tv}{\lVert u\rVert_2\lVert v\rVert_2}
$$

Softmax 将一组得分转换为归一化权重：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

实现中通常先减去该组得分的最大值，避免直接计算较大的指数。对真实类别 y，单样本交叉熵为 `−log(p_y)`；训练时通常直接调用接收 Logits 的稳定实现。

N-gram 语言模型近似为：

$$
P(w_t\mid w_1,\ldots,w_{t-1})\approx P(w_t\mid w_{t-n+1},\ldots,w_{t-1})
$$

其窗口之外的信息不直接进入这个条件概率。遇到未见过的组合，还需要平滑等方法，避免简单频率估计得到零概率。

多头注意力和 FFN 也可以写成紧凑的公式：

$$
\operatorname{head}_i=\operatorname{Attention}(XW_i^Q,XW_i^K,XW_i^V)
$$

$$
\operatorname{MultiHead}(X)=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
$$

$$
\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1+b_1)W_2+b_2
$$

这里将单个位置 x 写成行向量。若维度从 d 扩展到 f 再回到 d，则 `W1` 的形状为 `[d,f]`，`W2` 为 `[f,d]`；ReLU 将负值变为 0，非负值保持不变。注意力负责跨位置汇总，FFN 则在每个位置内部变换特征。

LSTM 的完整门控形式可以写成：

$$
\begin{aligned}
f_t&=\sigma(W_f[x_t;h_{t-1}]+b_f)\\
i_t&=\sigma(W_i[x_t;h_{t-1}]+b_i)\\
g_t&=\tanh(W_g[x_t;h_{t-1}]+b_g)\\
o_t&=\sigma(W_o[x_t;h_{t-1}]+b_o)\\
c_t&=f_t\odot c_{t-1}+i_t\odot g_t\\
h_t&=o_t\odot\tanh(c_t)
\end{aligned}
$$

`[x;h]` 表示拼接，`σ` 是 Sigmoid。PyTorch 将输入项和隐藏状态项分开存储权重及偏置，和这里合写的形式可以对应。

按 PyTorch 的 GRU 约定，候选状态与更新可写为：

$$
\begin{aligned}
r_t&=\sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr})\\
z_t&=\sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz})\\
n_t&=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn}))\\
h_t&=(1-z_t)\odot n_t+z_t\odot h_{t-1}
\end{aligned}
$$

LayerNorm 在一个 Token 的 d 维表示上计算：

$$
\mu=\frac1d\sum_{i=1}^d x_i,\qquad v=\frac1d\sum_{i=1}^d(x_i-\mu)^2
$$

$$
\operatorname{LN}(x_i)=\gamma_i\frac{x_i-\mu}{\sqrt{v+\epsilon}}+\beta_i
$$

`γ`、`β` 可学习，`ε` 防止除零。这里把 ε 加在方差之后、平方根之内，避免把标准差和方差混写。

原始 Transformer 的正余弦位置编码是：

$$
\operatorname{PE}(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
\operatorname{PE}(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

`pos` 是位置，`i` 指定一组偶数、奇数维。固定公式可以计算更远的位置，但这不保证模型在训练长度之外仍有可靠表现。

### B PyTorch 形状与参数

循环网络常用的构造参数包括 `input_size`、`hidden_size`、`num_layers`、`batch_first`、`bidirectional`、`dropout`，以及控制偏置、设备和数据类型的 `bias`、`device`、`dtype`。

`nn.RNN` 可以通过 `nonlinearity` 选择 `tanh` 或 `relu`；**`nn.LSTM` 没有这个构造参数**。LSTM 另有 `proj_size`。循环模块的 `dropout` 通常作用在堆叠层之间，单层时不能把它理解成自动对所有时间步增加循环 Dropout。

下面是可直接运行的形状检查：

```python
import torch
from torch import nn

x = torch.randn(2, 5, 16)  # batch=2, length=5, embedding=16
for cls in (nn.RNN, nn.LSTM, nn.GRU):
    layer = cls(16, 32, num_layers=2, batch_first=True, bidirectional=True)
    output, state = layer(x)
    h = state[0] if isinstance(state, tuple) else state
    print(cls.__name__, tuple(output.shape), tuple(h.shape))
```

三者均得到 `output=(2,5,64)`、`h=(4,2,32)`；LSTM 另返回同形状的 `c`。双向网络的 `h_n` 包含两个方向各自的最终状态，不等于简单取 `output` 的最后一个时间位置。

也可以传入初始状态 `h_0`，形状与相应的 `h_n` 一致；LSTM 另有 `c_0`，对应 `c_n`。不提供初始状态时，模块默认使用零状态。投影 LSTM 的 `h` 与 `c` 仍需分别按各自维度处理。

Transformer 的主要参数包括 `d_model`、`nhead`、`num_encoder_layers`、`num_decoder_layers`、`dim_feedforward`、`dropout` 和 `batch_first`。还可以选择 `activation`、`norm_first`、`layer_norm_eps`、`bias`、`device`、`dtype`，或通过 `custom_encoder`、`custom_decoder` 传入自定义模块。`dim_feedforward` 是独立参数，不会因为你改变 `d_model` 就自动变成它的四倍。

在 `batch_first=True` 时，编码器—解码器的主要契约是：

| 名称 | 形状或含义 |
| --- | --- |
| `src` | `[B, S, d_model]`，源序列表示 |
| `tgt` | `[B, T, d_model]`，右移后的目标序列表示 |
| `src_key_padding_mask` | `[B, S]`，屏蔽源补齐位置 |
| `tgt_key_padding_mask` | `[B, T]`，屏蔽目标补齐位置 |
| `src_mask` | 常为 `[S,S]`，控制源位置之间的可见关系，普通翻译通常无需设置 |
| `tgt_mask` | 常为 `[T,T]` 的因果掩码 |
| `memory_mask` | `[T,S]`，控制每个目标位置能读取哪些源位置，区别于按 batch 屏蔽 PAD |
| `memory` | 编码器输出，`[B,S,d_model]` |
| `memory_key_padding_mask` | `[B,S]`，交叉注意力屏蔽源补齐位置 |
| 核心解码器输出 | `[B,T,d_model]`，还需映射为 `[B,T,V_target]` 的词表得分 |

`src`、`tgt` 的批量大小与特征维度必须匹配，但序列长度 S、T 可以不同。因果掩码的主对角线保持可见，严格上三角才被屏蔽；布尔和浮点掩码的语义见第 9.5 节。PyTorch 还有 `src_is_causal`、`tgt_is_causal`、`memory_is_causal` 等提示参数，不要把错误的提示当作自动构造正确掩码的保证。完整契约见 [PyTorch 2.6 Transformer API](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Transformer.html)。

### C 读论文时容易碰到的模型规格和任务缩写

经典 BERT-base 是 12 层、768 隐藏维、12 头，参数量约 1.1 亿；BERT-large 是 24 层、1024 隐藏维、16 头，约 3.4 亿。实际参数量还随词表和任务头变化，所以这些是经典规格，而不是所有同名变体的精确文件大小。

| 缩写 | 大致在评估什么 |
| --- | --- |
| MNLI、RTE | 句子之间的蕴含等关系 |
| QQP、MRPC | 两个问题或句子是否表达相近意思 |
| QNLI | 问题与候选句子之间的回答关系 |
| STS-B | 语义相似度回归，输出连续得分 |
| SWAG | 从候选项中选择合理的后续内容 |
| SST-2 | 情感二分类 |
| CoLA | 句子的语法可接受性 |
| SQuAD | 基于给定段落的问答；具体版本决定是否包含无答案样本 |
| NER | 命名实体识别，是任务名称而非某一个固定数据集 |

这些名称有助于阅读模型卡，但公开基准成绩不能替代你的业务测试。

### D 示例与验证范围

配套材料包含八个可运行的演示入口：词袋基线、分词与 Word2Vec、NumPy 注意力、循环分类、三种翻译结构、BERT 输入输出、Datasets 处理和 BERT 微调。循环分类与翻译入口分别提供三个架构选项。

这次验证使用 Python 3.12 的 CPU 环境，运行了上述入口，检查了模型保存后重载、数据保存恢复及主要张量形状。小语料上的指标只用于展示评估链路。完整文件清单和运行方式见 [代码说明](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/README.md)，知识点覆盖及具体核验范围见 [知识点覆盖与验证说明](https://github.com/LMJian/LMJian.github.io/blob/main/examples/nlp-huggingface/coverage.md)。

正文中的十四张图均为配套绘制的教学图；注意力数值图与 NumPy 示例一致，位置编码图按附录公式计算，BERT 字段图使用指定 Tokenizer 的实际编码结果。网页已启用数学公式渲染。下载包中的正文需要支持 LaTeX 的 Markdown 查看器；移动离线正文时，请保留同级的 `assets` 文件夹。
