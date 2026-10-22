# NLP 与 Hugging Face 示例

所有命令都从本 README 所在目录执行。通过网站下载 ZIP 时，先解压并进入 `nlp-huggingface-demo/`；在博客仓库中，进入 `examples/nlp-huggingface/`。

示例使用人工编写的教学数据。评论数据分为 24 条训练、8 条验证、8 条测试；翻译数据分为 12 条训练、2 条验证、2 条测试。指标只展示计算过程，不能证明生产效果，也不能据此比较模型架构优劣。

## 安装

使用 Python 3.12，在独立虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。`requirements.txt` 固定直接依赖；`requirements-lock.txt` 记录本次复核环境的完整 Python 包版本，其中也包含绘图和文档检查工具；运行示例只需安装 `requirements.txt`。同一平台需要更严格复现时，可以用后者安装。其他操作系统的 PyTorch 安装方式仍以官方说明为准。

全部示例默认使用 CPU。首次安装依赖需要网络；前三组原理示例不需要下载模型。BERT 示例首次下载数百 MB 权重，保存本地副本和微调结果还需要额外磁盘空间。这里提供代码与小数据，不包含预训练权重或虚拟环境。

## 先运行不下载模型的部分

```bash
python examples/bow_baseline.py
python examples/tokenize_word2vec.py --out runs/word2vec
python examples/attention_numpy.py
python examples/sequence_classifier.py --arch rnn --out runs/rnn
python examples/sequence_classifier.py --arch lstm --out runs/lstm
python examples/sequence_classifier.py --arch gru --out runs/gru
python examples/translation.py --arch gru --out runs/translation-gru
python examples/translation.py --arch attention --out runs/translation-attention
python examples/translation.py --arch transformer --out runs/translation-transformer
```

Word2Vec 的语料非常小，特意用 `sample=0` 关闭高频词下采样，避免几乎没有可训练的上下文；已检查向量确实发生更新。得到的近邻和相似度仍不具有可靠语义。翻译例子默认训练 80 个 epoch，但每个 epoch 只有一个很小的 batch；它用于观察结构、掩码和生成流程，不能学习通用翻译能力。

## 再运行 Hugging Face 部分

```bash
python examples/hf_inspect.py --out runs/bert-local
python examples/datasets_demo.py --model runs/bert-local --out runs/datasets
python examples/hf_finetune.py --model runs/bert-local --epochs 1 --out runs/bert-review
```

第一条命令从 Hub 加载 `google-bert/bert-base-chinese`，默认固定 revision 为 `8f23c25b06e129b6c986331a13d8d025a92cf0ea`，检查输入输出，保存后再次从本地加载并比较结果。公开模型通常无需登录；是否需要权限，以实际仓库状态为准。

后两条命令复用完整的本地模型目录。若要强制离线，可以在运行前设置 `HF_HUB_OFFLINE=1`。这要求第一条命令已成功完成；离线开关不会补齐缺失文件。

微调脚本默认训练整个 BERT。可以添加 `--freeze-backbone` 只训练分类头；它会使用不同的学习率，并关闭冻结主干的 Dropout。脚本只针对教学规模设计，未实现混合精度、分布式、断点续训或部署服务。

再次运行时，`--out` 所指目录中的同名结果可能被覆盖。想保留不同实验，请使用不同输出目录。

## 文件分工

| 文件 | 内容 |
| --- | --- |
| `reviews.csv` | 自包含评论数据及固定 split |
| `common.py` | 随机种子、数据读取、Accuracy 与 Macro-F1 |
| `bow_baseline.py` | 字符 TF-IDF 与逻辑回归基线 |
| `tokenize_word2vec.py` | jieba 三种模式、用户词典、Word2Vec 训练与持久化、Embedding 行对齐 |
| `attention_numpy.py` | 数值注意力、行归一化及因果掩码 |
| `sequence_classifier.py` | 可切换的 RNN、LSTM、GRU 分类完整流程 |
| `translation.py` | GRU Seq2Seq、带注意力版本、Transformer，以及 BLEU 演示 |
| `hf_inspect.py` | 真正的 BERT 预训练权重加载、Tokenizer API、输入输出及保存重载 |
| `datasets_demo.py` | CSV 加载、清洗、分层划分演示、map、持久化、CSV/JSON 逐条重载核对与动态补齐 |
| `hf_finetune.py` | BERT 微调、验证集选模、测试、保存重载与 pipeline 推理 |

## 验证边界

这些入口都在 Python 3.12 CPU 环境执行过。验证还覆盖了序列模型追加补齐不改变有效预测，以及改变目标未来 Token 不影响前面位置的预测。

BERT 仅使用 24 条样本训练一轮，本次测试准确率为 0.50；词袋基线为 0.75。它们只展示真实运行结果，不是推荐模型的依据。随机数、底层计算库和平台差异可能导致数值变化。

评论分类阈值固定为教学用的 0.5。翻译指标使用 NLTK `corpus_bleu`、均匀的一至四元权重、`SmoothingFunction().method1`，返回 0 到 1；两条短测试句上的该分数不适合比较模型质量。

需要排查问题时，先检查当前解释器、依赖版本、模型目录完整性、输入字段和张量形状。网络错误、访问权限错误与模型代码错误是不同问题；不要通过关闭 TLS 证书校验来掩盖下载失败。
