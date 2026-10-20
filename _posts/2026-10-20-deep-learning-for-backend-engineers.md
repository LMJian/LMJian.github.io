---
title: "写给后端工程师的深度学习入门：从张量到能训练的神经网络"
description: "面向业务后端开发者，用 18 张图理解张量、神经网络、反向传播和 CNN，并用 PyTorch 完成线性回归、数字识别、房价回归与服装分类。附可运行代码与下载包。"
date: 2026-10-20 10:00:00 +0800
categories: [人工智能, 深度学习]
tags: [深度学习, pytorch, 神经网络, CNN, python, 后端开发]
math: true
toc: true
media_subpath: /assets/img/posts/deep-learning
---

假设你要给一个业务系统增加“识别用户上传图片”的能力。接口、鉴权、对象存储、异步任务都很熟悉，真正难的是中间那个函数：输入一张图片，输出它属于哪个类别。

你可以试着写规则，但同一个数字换一种笔迹、同一件衣服换一个角度，规则就可能失效。深度学习提供了另一种做法：先定义一个带有大量可调整参数的计算过程，再用带答案的数据，逐步把参数调到能完成任务的状态。

这篇文章要讲清楚的，就是这个“逐步调好”的过程。读完后，你应该能够解释一段 PyTorch 训练代码在做什么，运行一个小模型，并知道怎样判断它是否真的学到了有用的规律。

内容以监督学习为主：每个训练样本都有对应的正确答案。我们先用线性回归看清训练的基本动作，再把同一套方法用于手写数字、房价回归和服装分类。你不需要先学完整本高等数学；遇到公式时，重点看变量代表什么，以及它如何对应代码。

可以分三次读：

- **建立直觉**：[基本概念](#what) → [环境](#environment) → [张量](#tensor) → [神经网络](#network) → [损失](#loss) → [反向传播](#backprop)。
- **跑通训练**：[训练循环](#training) → [优化器](#optimization) → [初始化与泛化](#initialization) → [数字识别](#mnist) → [房价回归](#housing)。
- **理解图像模型**：[CNN 与服装分类](#cnn) → [经典结构](#architectures) → [业务接入](#engineering) → [学习建议](#next)。[张量 API](#tensor-api)、[优化器公式](#optimizer-math)和[导数](#derivatives)放在附录。

<a id="download"></a>

[下载配套代码与图解（ZIP）]({{ '/assets/downloads/deep-learning-demo.zip' | relative_url }})。解压后进入 `deep-learning-demo/`，再运行本文命令：`examples/` 包含 5 个完整脚本和固定版本的依赖，`assets/` 包含 18 张图及数据许可说明。图中的示意曲线与实际实验结果会明确区分。

如果想先用业务表格数据熟悉特征、标签和评估，可以阅读[机器学习入门：从业务问题到可运行的预测模型]({% post_url 2026-10-13-machine-learning-for-backend-engineers %})。

<a id="what"></a>

## 1 深度学习到底在做什么

### 1.1 从写规则到学习一个函数

普通业务代码通常是“人写规则，程序执行规则”。例如，订单超过 30 分钟未支付，就进入关闭流程。

机器学习适合另一类问题：规则很难逐条写全，但我们有过去的输入和结果。例如，给定一张图片，判断它是裤子还是鞋；给定房屋面积、房龄和所在区域，估计一个价格。

用一个公式表示：

$$
\hat y=f_\theta(x)
$$

- $$x$$ 是输入，也叫**特征**。它可以是一组业务字段，也可以是图片的像素。
- $$y$$ 是正确答案，也叫**标签**。分类任务中它是类别，回归任务中它是一个数值。
- $$\hat y$$ 是预测结果。
- $$f$$ 是模型的计算结构。
- $$\theta$$ 是模型的参数，例如后面会看到的权重和偏置。

**训练**负责从数据中调整 $$\theta$$；**推理**使用已经确定的参数处理新输入。调用预测接口通常属于推理，接口不会因为多处理了一次请求就自动变聪明。

![训练负责学习参数 推理负责处理新输入](01-learning-system.png){: width="1978" height="999" }

*图 1：训练有输入和标签，推理只有待处理的输入。数据处理规则和模型参数需要一起交付。*

### 1.2 深度学习与机器学习是什么关系

人工智能是较大的领域；机器学习是其中一类从数据中学习的方法；深度学习则是机器学习的一部分，主要使用多层神经网络学习表示和预测。

这里的“深”说的是计算层次。图像模型的浅层可能响应边缘和纹理，更深的层组合这些信息，形成更适合分类的表示。这些表示是训练得到的，不要求开发者逐项定义“衣领应该怎么检测”。

神经网络的名字受到生物神经系统启发。生物神经元接收信号、整合信号，再向其他神经元传递；人工神经元借用了这种连接与传递的想法，但它执行的是数值计算，并不是对人脑的完整模拟。

大语言模型也属于深度学习模型，但“学深度学习”不等于“调用大模型 API”。张量、损失、自动微分和训练流程，是理解更复杂模型的基础。

### 1.3 为什么它重要 又有哪些代价

深度学习擅长从复杂数据中学习多层次的表示，因此经常用于图像识别、语音识别、文本处理和推荐等任务。对应到业务系统，就是图片分类、客服语音转写、文本分流、个性化推荐、工业视觉质检等能力。风险识别、医学影像辅助分析、驾驶感知、遥感与安防影像分析，也会用到类似技术，但各自需要领域数据和专门的评估。

它的代价同样具体：训练数据需要采集和整理，标签可能出错；模型训练需要时间与计算资源；输出不总是容易解释；业务环境变化后，模型还可能退化。

所以，当规则足够清晰时，继续写规则通常很好。面对规模不大的业务表格数据，也应该先建立线性模型、树模型等基线。**基线**就是一个简单、可靠、方便比较的起点。神经网络是否值得用，要由效果、成本和维护难度共同决定。

<a id="environment"></a>

## 2 准备一个能运行的 PyTorch 环境

PyTorch 提供了三项在这里最需要的能力：

1. 用张量进行数值计算，并把计算放到 CPU 或合适的加速设备上。
2. 用 `torch.nn` 组合神经网络层。
3. 自动计算梯度，再用优化器更新参数。

它源于 Torch 生态，主要使用 Python 编写模型逻辑，并由底层实现完成大量数值计算。初学时，把它看成“支持自动求导的数组计算库，加上一组模型组件”就足够了。

本文示例使用 Python 3.12、PyTorch 2.9.0 和 torchvision 0.24.0。这是一组用于复现实验的版本，**不是要求读者使用最新版本**。完整依赖见[配套下载包](#download)中的 `examples/requirements.txt`。

先进入配套目录，创建独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
python examples/01_tensor_basics.py
```

Windows PowerShell 中，用 `.venv\Scripts\Activate.ps1` 激活环境。若使用 Conda，也可以创建 Python 3.12 环境；关键是终端、IDE 和 Notebook 使用同一个解释器。Notebook 用户可在这个环境中安装 `jupyter`，再从这个环境启动。

上面的安装命令使用平台默认的 PyPI 包。Linux 或 Windows 如果明确只需要 CPU 包，可以先按 [PyTorch 历史版本安装页](https://pytorch.org/get-started/previous-versions/) 为指定版本选择 CPU 命令；使用 NVIDIA GPU 或 AMD ROCm 时，则按 [官方安装选择器](https://pytorch.org/get-started/locally/) 和对应硬件支持情况选择安装方式，不要机械复制其他机器的命令。

### 2.1 GPU 环境里不要混淆三个版本

以 NVIDIA 为例：

- **驱动**让操作系统能够使用 GPU。`nvidia-smi` 显示的 CUDA 版本反映驱动支持能力，不代表本机安装了那个版本的开发工具包。
- **CUDA Toolkit**包含编译器等开发工具。`nvcc --version` 查看的是这一层。
- **PyTorch 的 CUDA 构建版本**对应它所使用的运行时依赖，代码中可通过 `torch.version.cuda` 查看。

使用官方预编译的 PyTorch 包时，通常不需要为了运行模型另装完整 Toolkit；编译源码或自定义 CUDA 扩展时才经常需要它。仍然需要兼容的驱动和受该构建支持的 GPU。显卡计算能力下限会随构建版本变化，不应长期记成一个固定数字。关于运行时与 Toolkit 的区别，可参考 [PyTorch 维护者的说明](https://discuss.pytorch.org/t/pytorch-version-for-cuda-12-9/221651/2)。

Apple Silicon 的 MPS 是另一种加速后端，不使用 CUDA；AMD ROCm 也有自己的兼容范围。本文所有示例默认使用 CPU，可以先把学习流程跑通，再考虑加速。

下面的检查可以独立运行：

```python
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("selected device:", device)
print(torch.ones(2, 3, device=device).sum().item())
```

离线安装时，需要提前下载与 Python 版本、操作系统、CPU 架构匹配的 wheel 及其依赖，同时检查 `torch` 与 `torchvision` 的版本配对。不要只拿到一个文件名中带 `cuXXX` 的 wheel 就认为环境已经匹配。

<a id="tensor"></a>

## 3 张量是带有形状 类型和设备的数据

### 3.1 先理解 shape

在本文的语境中，**张量 Tensor**可以先理解成多维数组：

| 数据 | 例子 | 常见形状 |
|---|---|---|
| 标量 | 一个损失值 | `[]`，零维 |
| 向量 | 一条样本的 8 个特征 | `[8]` |
| 矩阵 | 32 条样本，每条 8 个特征 | `[32, 8]` |
| 单张灰度图 | 1 个通道，28×28 个像素 | `[1, 28, 28]` |
| 一批彩色图 | 32 张 RGB 图片 | `[32, 3, H, W]` |

**维度的个数**与**某个维度的大小**是两件事。`[32, 3, 28, 28]` 是四维张量，第二个维度的大小是 3。

对于后端工程师，`shape` 很像接口的数据契约。每层都要求输入满足某种形状，再输出另一种形状。很多初学错误并不是算法错误，而是把 `[N, D]`、`[D, N]`、`[N]` 和 `[N, 1]` 混用了。

![业务表格和图片如何表示成张量](02-tensor-shapes.png){: width="1978" height="921" }

*图 2：`N` 表示一批样本的数量。图像常用 `NCHW`，即批次、通道、高、宽。*

除了 `shape`，还要关注：

- `dtype`：元素类型。本文输入和模型参数主要用 `float32`；整数类别标签用 `int64`，也叫 `torch.long`。
- `device`：数据位于 CPU、CUDA 还是 MPS。
- `requires_grad`：是否需要自动求导。通常模型参数需要，输入样本和标签不需要。

模型与参与计算的输入应放在兼容的设备上，类型也要匹配。`tensor.to(...)` 的返回值需要接住；不能假设调用后原变量已经变了。

### 3.2 两种乘法解决不同问题

`x * y` 是逐元素乘法，又叫**哈达玛积**。两个张量的对应元素相乘，或先按广播规则对齐。

`x @ w` 是矩阵乘法。若 `x` 是 `[N, D]`，`w` 是 `[D, H]`，结果就是 `[N, H]`。这会把每条样本的 `D` 个输入组合成 `H` 个输出，正好对应全连接层的主要计算。

**广播 broadcasting**允许部分不同形状的张量做逐元素运算：从最右侧维度开始比较，对应大小相同、某一侧大小为 1，或缺少该维度时，可以对齐。例如 `[32, 8] + [8]` 可以给每条样本加同一组偏置。

广播也会掩盖错误。回归预测是 `[32, 1]`，标签却是 `[32]` 时，相减可能得到 `[32, 32]`。计算没有报错，含义已经错了。

下面的脚本可以独立运行，覆盖计算、索引、拼接和内存共享：

```python
import numpy as np
import torch

torch.manual_seed(42)
x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
w = torch.ones(3, 2)
print("x:", x.tolist())
print("elementwise:", (x * x).tolist())
print("matrix product:", (x @ w).tolist())
print("column mean:", x.mean(dim=0).tolist())
print("row mean:", x.mean(dim=1, keepdim=True).tolist())
print("paired indexing:", x[[0, 1], [1, 2]].tolist())
print("boolean indexing:", x[x > 2].tolist())
print("transpose:", tuple(x.transpose(0, 1).shape))
print("cat:", tuple(torch.cat([x, x], dim=0).shape))
print("stack:", tuple(torch.stack([x, x], dim=0).shape))
print("unsqueeze:", tuple(x.unsqueeze(1).shape))

a = np.array([1.0, 2.0], dtype=np.float32)
shared = torch.from_numpy(a)
copied = torch.tensor(a)
a[0] = 9.0
print("shared / copied:", shared.tolist(), copied.tolist())
assert shared[0].item() == 9.0 and copied[0].item() == 1.0

t = x.transpose(0, 1)
print("contiguous:", t.is_contiguous())
print("reshape:", t.reshape(-1).tolist())
print("contiguous then view:", t.contiguous().view(-1).tolist())
assert torch.equal(t.reshape(-1), t.contiguous().view(-1))
```

### 3.3 改形状与交换维度不能混用

`reshape` 重新组织元素的形状，`transpose` 和 `permute` 改变轴的排列。

假设读入的彩色图片是 `[H, W, C]`，要交给通常使用 `[C, H, W]` 的图像模型，应使用 `permute(2, 0, 1)`。直接 `reshape(C, H, W)` 会把元素按不同方式分组，不能代替通道轴的交换。

`view()` 与原张量共享存储，但目标形状必须与原来的大小和 **stride，内存步幅**兼容。连续张量通常最容易满足这个条件；不能笼统地说“任何非连续张量都不能 view”。不确定时可以用 `reshape()`，它可能返回视图，也可能复制。需要明确连续布局时，用 `contiguous().view(...)`。这些边界见 [Tensor.view 文档](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.view.html)。

再记住几个容易踩的地方：

- `squeeze()` 会删除大小为 1 的维度。如果批量刚好是 1，它也可能删掉批次轴；回归输出通常写成 `squeeze(-1)` 更明确。
- `cat` 沿已有维度连接，`stack` 新建一个维度后堆叠。
- `torch.from_numpy(a)` 通常共享数组内存，`torch.tensor(a)` 会复制。转回 NumPy 时，常用 `x.detach().cpu().numpy()`；如果需要独立副本，再 `.copy()`。
- `add_()` 等带尾部下划线的方法通常原地修改数据。对需要梯度的张量，原地操作可能破坏求导所需的值。
- `torch.empty(...)` 和按尺寸调用的旧式 `torch.Tensor(...)` 不会替你填好有效数据。看到“随机数一样的值”，不代表它进行了随机采样。

更多创建、统计和索引 API 放在[附录 A](#tensor-api)，不用一次背完。

<a id="network"></a>

## 4 神经网络是一串可以学习的计算

### 4.1 一个神经元做了什么

先看最小形式：

$$
z=w_1x_1+w_2x_2+\cdots+w_dx_d+b,\qquad a=\phi(z)
$$

权重 $$w$$ 决定输入如何参与组合，偏置 $$b$$ 提供一个可学习的平移量，激活函数 $$\phi$$ 再对结果做变换。这里的“激活”不用想得很神秘，它就是一个函数调用。

把多个神经元放在一起，就得到一层；把层连接起来，就得到网络。输入层接收特征，隐藏层产生中间表示，输出层产生最终预测所需的数值。**前向传播**就是按依赖顺序把这些计算执行一遍。

### 4.2 全连接层如何对应矩阵乘法

每个输出都连接到上一层的全部输入，称为全连接层，也叫 Dense 层。在 PyTorch 中通常用 `nn.Linear`：

```python
import torch
from torch import nn

layer = nn.Linear(3, 4)
x = torch.randn(5, 3)
y = layer(x)
print(y.shape)                  # torch.Size([5, 4])
print(layer.weight.shape)       # torch.Size([4, 3])
print(layer.bias.shape)         # torch.Size([4])
print(sum(p.numel() for p in layer.parameters()))  # 3*4+4 = 16
```

数学上可以约定 $$Y=XW+b$$，让 $$W$$ 的形状是 `[输入数, 输出数]`。PyTorch 的 `Linear.weight` 则存成 `[输出数, 输入数]`，实际计算是 $$XW^\top+b$$。两个写法等价，阅读代码时要统一约定。

矩阵乘法再加偏置，数学上叫**仿射变换**。这里还有一个重要限制：连续堆叠仿射变换，却没有非线性激活，最终仍然等价于一次仿射变换：

$$
(XW_1+b_1)W_2+b_2
=X(W_1W_2)+(b_1W_2+b_2)
$$

因此，增加层数本身不足以让模型表达复杂的非线性关系。

<a id="activations"></a>

### 4.3 激活函数让模型能够表达非线性

![Sigmoid Tanh ReLU 及其导数](03-activations.png){: width="2039" height="1126" }

*图 3：上排看输出怎样变化，下排看梯度怎样传递。Sigmoid 和 Tanh 在两端趋于平坦。*

初学时先理解三个函数：

| 函数 | 公式与输出范围 | 如何理解 | 主要限制 |
|---|---|---|---|
| Sigmoid | $$\sigma(z)=1/(1+e^{-z})$$，输出在 `(0,1)` | 将一个分数转成二分类概率形式 | 两端饱和，导数接近 0；输出不以 0 为中心 |
| Tanh | $$\tanh(z)$$，输出在 `(-1,1)` | 类似居中的 Sigmoid | 两端仍然会饱和 |
| ReLU | $$\max(0,z)$$ | 正数保留，负数变成 0 | 持续处于负半轴的单元可能学不动 |

ReLU 的正半轴导数是 1，有助于缓解饱和带来的梯度衰减，但它**不能保证整个网络不会梯度消失**。负半轴导数为 0，其他层的权重也会影响梯度传播。在 0 点它不可导，PyTorch 采用的反向传播值为 0。

阶跃函数则直接输出 0 或 1。它在非阈值位置导数为 0，在阈值处不可导，不适合直接用于常规梯度训练。

多分类还会用到 **Softmax**。它把一组分数转换成总和为 1 的概率：

$$
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}
$$

Softmax 作用于一组类别分数，而不是分别把每个元素转换成独立概率。输入为 `[N, C]` 时，`dim=1` 表示在类别维度归一化。实际计算会使用数值稳定的方法，例如先减去这组分数的最大值。

这里的原始分数 $$z$$ 叫 **logits**。它们可以为负，也不要求相加等于 1。下一节会说明为什么训练时经常直接把 logits 交给损失函数。

其他激活函数也值得认识，但暂时不必逐个试一遍：

| 函数 | 主要变化 | 常见 API |
|---|---|---|
| Identity | $$f(z)=z$$，原样输出 | `nn.Identity()` |
| Leaky ReLU | 负半轴保留固定小斜率 $$\alpha$$ | `nn.LeakyReLU(0.01)` |
| PReLU | 让负半轴斜率也成为可学习参数 | `nn.PReLU()` |
| RReLU | 训练时在给定区间均匀采样负半轴斜率，评估时使用区间中点 | `nn.RReLU()` |
| ELU | 正半轴为 $$z$$，负半轴为 $$\alpha(e^z-1)$$ | `nn.ELU()` |
| SiLU | $$z\sigma(z)$$；相当于参数为 1 的 Swish | `nn.SiLU()` |
| Softplus | $$\log(1+e^z)$$，平滑的正值变换 | `nn.Softplus()` |

本文的小型前馈网络用 ReLU 作为隐藏层起点。输出层则必须与任务、损失函数配合：回归可以直接输出数值，二分类可以输出一个 logit，多分类可以输出每个类别的 logit。不要看到“分类”就无条件在模型末尾加 Softmax。

这些激活既可以写成 `torch.relu(x)`、`torch.sigmoid(x)` 等函数调用，也可以用 `nn.ReLU()`、`nn.Sigmoid()` 等模块放入 `Sequential`，对应的数学操作相同。

### 4.4 用 Module 把层组织起来

`nn.Module` 类似一个带状态的计算组件。`__init__` 中注册子层，`forward` 描述输入如何流动。注册到模块上的参数会出现在 `parameters()` 中，供优化器更新。

```python
import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.ReLU(),
            nn.Linear(4, 2),
        )

    def forward(self, x):
        return self.layers(x)  # 输出 logits

model = Classifier()
print(model)
print(model(torch.randn(10, 3)).shape)  # [10, 2]
print(sum(p.numel() for p in model.parameters()))  # 46
for name, parameter in model.named_parameters():
    print(name, tuple(parameter.shape))
```

纯串行结构可以只用 `nn.Sequential`。有分支、跳跃连接或多路输入时，自定义 `forward` 更方便。调用时写 `model(x)`，让 `Module` 的调用机制正常处理钩子等逻辑，不直接调用 `model.forward(x)`。

查看结构先用 `print(model)`，统计参数用 `numel()` 就够了；需要逐层形状报告时，也可以使用模型摘要工具，例如 `torchsummary`，但它不是训练所必需的依赖。

<a id="forward-steps"></a>

### 4.5 把一次前向传播逐层拆开

`model(x)` 只是把多个步骤封装成一次调用。换一个更小的 `2 → 3 → 2 → 2` 网络，设两个隐藏层都用 ReLU，输出层直接给出 logits：

$$
H_1=\operatorname{ReLU}(XW_1+b_1),\qquad
H_2=\operatorname{ReLU}(H_1W_2+b_2),\qquad
Z=H_2W_3+b_3
$$

`H` 表示隐藏层输出，`Z` 表示最后的分数。对一批输入，各层形状依次为 `[N,2] → [N,3] → [N,2] → [N,2]`。输入节点只是接收数据，通常不计入可学习层数；这里有三个带权重的层。

![从输入到两个隐藏层再到输出的完整前向传播](13-dense-forward.png){: width="1978" height="1159" }

*图 4：连线代表权重，偏置写在各层公式中。固定一组便于手算的参数后，输入 `[1,2]` 最终得到分数 `[3,-2]`。*

下面独立运行一次，观察每层怎样把上一层的结果继续加工。权重采用本节的数学约定 `[输入数,输出数]`：

```python
import torch

x = torch.tensor([[1.0, 2.0]])
w1 = torch.tensor([[1., 0., -1.], [0., 1., 1.]])
b1 = torch.tensor([0., 0., -3.])
w2 = torch.tensor([[1., -1.], [1., 1.], [0., 1.]])
b2 = torch.tensor([0., -2.])
w3 = torch.tensor([[1., -1.], [1., 1.]])
b3 = torch.tensor([0., 1.])

z1 = x @ w1 + b1
h1 = z1.relu()
z2 = h1 @ w2 + b2
h2 = z2.relu()
logits = h2 @ w3 + b3
print("layer 1:", z1.tolist(), "->", h1.tolist())  # [1,2,-2] -> [1,2,0]
print("layer 2:", z2.tolist(), "->", h2.tolist())  # [3,-1] -> [3,0]
print("logits:", logits.tolist())                 # [[3,-2]]
torch.testing.assert_close(logits, torch.tensor([[3., -2.]]))
```

这组参数只是为了演示计算，没有训练过。它解释了网络怎样运行，不能据此判断模型会不会分类。

<a id="loss"></a>

## 5 损失函数把预测变成可优化的目标

模型算出预测后，还需要知道“错了多少”。**损失函数 loss**把预测与标签之间的差异变成一个数，训练过程尝试降低这个数。

它与业务指标不一定相同。例如分类准确率只区分答对、答错；交叉熵还会区分“答对但犹豫”和“答错且很自信”。后者更适合给梯度训练提供连续的反馈。

### 5.1 回归中的 MAE MSE 和 Smooth L1

令 $$e_i=\hat y_i-y_i$$，对 $$N$$ 个单输出样本：

$$
\operatorname{MAE}=\frac1N\sum_i|e_i|,
\qquad
\operatorname{MSE}=\frac1N\sum_i e_i^2
$$

MAE 对应 `nn.L1Loss()`，大误差按线性增长；MSE 对应 `nn.MSELoss()`，大误差会被平方放大，因此对异常值更敏感。MSE 不意味着必然发生梯度爆炸，但目标尺度过大或异常值过多时，确实可能产生很大的梯度。

`SmoothL1Loss(beta=1)` 在小误差处使用平滑的二次函数，在大误差处改用线性增长：

$$
\ell(e)=
\begin{cases}
\frac12e^2,& |e|<1\\
|e|-\frac12,& |e|\ge1
\end{cases}
$$

一般 `beta > 0` 时，前半段为 $$e^2/(2\beta)$$，分界点为 $$\lvert e\rvert=\beta$$，后半段为 $$\lvert e\rvert-\beta/2$$。不要把不同 `beta` 的 Smooth L1 与 Huber Loss 当成数值完全相同的函数。

![三种回归损失对误差的响应](04-regression-losses.png){: width="1698" height="898" }

*图 5：横轴是预测误差，纵轴是单个样本的损失。MSE 的惩罚随误差增长得更快。*

PyTorch 默认 `reduction="mean"` 通常对参与计算的元素取均值。多输出回归时，分母还包括输出元素数量，不能把所有场景都理解成“只除以 batch size”。

### 5.2 分类为什么常用交叉熵

二分类中，标签 $$y_i$$ 为 0 或 1，模型给出属于正类的概率 $$p_i$$：

$$
L=-\frac1N\sum_i\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right]
$$

如果真实标签为 1，模型越接近输出 1，损失越小；如果它非常自信地输出接近 0，损失会很大。

单标签多分类中，每个样本只有一个正确类别。用 one-hot 向量表示标签，即正确类别位置为 1、其余位置为 0：

$$
L=-\frac1N\sum_i\sum_{c=1}^{C}y_{ic}\log p_{ic}
$$

对于硬标签，只有正确类别对应的一项保留下来，也就是 $$-\log p_{i,\text{正确类别}}$$。

实际编码时，先核对下面这张表：

| 任务 | 模型输出 | 标签 | 损失函数 | 推理时的转换 |
|---|---|---|---|---|
| 单输出回归 | `[N,1]` 浮点值 | 同形状浮点值 | `MSELoss` / `L1Loss` / `SmoothL1Loss` | 按目标定义还原尺度 |
| 二分类 | `[N,1]` 原始 logit | `[N,1]` 浮点 0/1 | `BCEWithLogitsLoss` | `sigmoid` 后选择阈值 |
| 单标签多分类 | `[N,C]` logits | `[N]`，`long`，范围 `0…C-1` | `CrossEntropyLoss` | `argmax` 或 `softmax` |
| 多标签分类 | `[N,C]` 独立 logits | `[N,C]` 浮点 0/1 | `BCEWithLogitsLoss` | 每类分别 `sigmoid` |

`BCEWithLogitsLoss` 把 Sigmoid 与二元交叉熵合在一起，用更稳定的方式计算。若使用 `BCELoss`，输入必须已经是概率。两者不能混用输入格式。[BCEWithLogitsLoss 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html)

`CrossEntropyLoss` 接收 logits，内部完成相应的 LogSoftmax 与负对数似然计算，不要再提前套一层 Softmax。它也支持与 logits 同形状的浮点软标签，例如各类别的目标概率；这种情况下需要自己确保概率分布有效。[CrossEntropyLoss 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)

一个可运行的小检查：

```python
import torch
from torch import nn

binary_logits = torch.tensor([[2.0], [-1.0]])
binary_labels = torch.tensor([[1.0], [0.0]])
print(nn.BCEWithLogitsLoss()(binary_logits, binary_labels))

class_logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.8, 0.0]])
class_labels = torch.tensor([0, 1], dtype=torch.long)
print(nn.CrossEntropyLoss()(class_logits, class_labels))
print(class_logits.softmax(dim=1))
```

<a id="backprop"></a>

## 6 模型怎样通过反向传播学习

### 6.1 梯度回答参数应该往哪边调

假设损失为 $$L(\theta)$$，梯度 $$\nabla_\theta L$$ 表示损失对各个参数的局部变化率。在足够小的步长下，沿负梯度方向移动，有助于降低当前损失：

$$
\theta \leftarrow \theta-\eta\nabla_\theta L
$$

$$\eta$$ 是**学习率 learning rate**，决定一步走多大。太小，进展慢；太大，可能跨过合适区域，甚至让训练发散。局部下降方向不代表任意步长都能下降，也不保证找到全局最优解。

**反向传播**负责计算梯度，**优化器**负责用梯度更新参数。这是两个不同动作。

### 6.2 从一个能手算的例子开始

设：

$$
x=2,\quad w=3,\quad b=1,\quad y=10
$$

前向计算得到 $$\hat y=wx+b=7$$，平方损失 $$L=(7-10)^2=9$$。

想知道调整 $$w$$ 会怎样影响损失，需要沿着计算依赖拆开：

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial\hat y}
\frac{\partial\hat y}{\partial w}
=2(\hat y-y)x=-12
$$

同理，$$\partial L/\partial b=2(\hat y-y)=-6$$。这是**链式法则**：把一条计算路径上各段的变化率相乘；一个变量影响结果的路径不止一条时，再把这些路径的贡献相加。

![前向计算数值 反向传递梯度](05-backpropagation.png){: width="1978" height="999" }

*图 6：前向得到预测与损失，反向沿依赖关系计算参数梯度。这个例子中梯度为负，梯度下降会增大权重与偏置。*

加法节点的局部导数是 1，因此把上游梯度传给各输入；乘法节点 $$u=ab$$ 对 $$a$$ 的局部导数是 $$b$$，对 $$b$$ 的局部导数是 $$a$$。大网络就是这些局部规则的组合。

### 6.3 自动微分替我们执行链式法则

PyTorch 在通常的即时执行模式下，随着前向计算记录求导所需的关系，建立计算图。只要相关运算可以求导，就不必为每种网络手写全部导数。

```python
import torch

x = torch.tensor(2.0)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(10.0)
prediction = w * x + b
loss = (prediction - y).square()
loss.backward()
print("prediction / loss:", prediction.item(), loss.item())
print("dw / db:", w.grad.item(), b.grad.item())
assert w.grad.item() == -12.0 and b.grad.item() == -6.0

# backward only computes gradients; an optimizer changes parameters.
optimizer = torch.optim.SGD([w, b], lr=0.1)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print("updated w / b:", round(w.item(), 2), round(b.item(), 2))

u = torch.tensor(2.0, requires_grad=True)
v = (u * u).detach()
(v * u).backward()
print("detached gradient:", u.grad.item())  # v is treated as a constant: 4
assert u.grad.item() == 4.0
```

主要输出如下：

```text
prediction / loss: 7.0 9.0
dw / db: -12.0 -6.0
updated w / b: 4.2 1.6
detached gradient: 4.0
```

这段代码还展示了 `detach()`：它让新张量脱离原计算图，但通常仍与原张量共享存储。最后的例子中，`v=u*u` 的数值仍是 4，可它被当作常量，所以 `v*u` 对 `u` 的导数是 4，而不是 $$3u^2=12$$。

需要认识的属性有：

- `is_leaf`：是否是按自动微分规则定义的叶子张量。直接创建且需要梯度的模型参数通常是叶子。
- `grad_fn`：非叶子结果是通过什么操作生成的。
- `.grad`：反向传播后保存的梯度。叶子参数默认会累积梯度，非叶子张量若要保留梯度，需要调用 `retain_grad()`。

普通训练不需要用 `.data` 绕过自动微分机制来改参数；初始化可以用 `nn.init`，手动更新可在 `torch.no_grad()` 中进行。需要梯度的叶子张量在梯度记录开启时，不能随意做原地修改。

默认一次反向后，图中用于再次求导的一些中间结果会被释放。常规训练在下一批数据上重新前向、重新建图即可，不要为解决普通循环错误而随手加 `retain_graph=True`。自动微分机制见 [PyTorch 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

全连接层、Sigmoid、ReLU 和 Softmax 加交叉熵的关键梯度，在[附录 C](#derivatives)中给出了推导和形状检查。

<a id="training"></a>

## 7 把训练组织成一个正确的循环

### 7.1 Dataset 和 DataLoader 各负责什么

`Dataset` 描述“有哪些样本、怎样取出一条样本”。自定义数据集通常实现 `__len__` 和 `__getitem__`，后者返回一条输入及其标签。已有张量时，可以用 `TensorDataset(x, y)`。

`DataLoader` 把样本组成批次，并支持打乱、并行读取等工作。它与 Dataset 的关系，有点像批处理执行器与数据访问接口的关系。

下面的例子可以独立运行：

```python
import torch
from torch.utils.data import Dataset, DataLoader

class TinyDataset(Dataset):
    def __init__(self):
        self.x = torch.arange(10, dtype=torch.float32).unsqueeze(1)
        self.y = 2 * self.x + 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

for x, y in DataLoader(TinyDataset(), batch_size=4, shuffle=True):
    print(x.shape, y.shape)  # 两批 [4,1]，最后一批 [2,1]
```

几个术语放在同一个场景下就容易理解：

- **Batch size**：一次送进模型的样本数量。
- **Iteration**：通常指处理一个批次的训练迭代。在本文代码中，每批计算一次梯度并更新一次参数。
- **Epoch**：完整遍历一次训练集。

2000 个样本、批量大小 64、保留最后不足一批的数据时，每轮有 $$\lceil2000/64\rceil=32$$ 个批次，最后一批 16 个样本。训练 10 轮是 320 次参数更新。若启用 `drop_last=True`，或者进行梯度累积，迭代数与更新数需要按实际设置重新计算。

严格说，随机梯度下降可指每次用一个样本估计梯度；工程里也经常把小批量训练称为 SGD。`torch.optim.SGD` 本身不会替你抽样，使用哪些数据由 DataLoader 和训练循环决定。

一次使用全体训练样本计算梯度，叫**全批量梯度下降**；一次使用一个样本，梯度波动通常更大；小批量在计算效率、内存占用和梯度波动之间折中。更大的 batch 并不自动带来更好的泛化效果。

### 7.2 训练集 验证集 测试集不能互相代替

**训练集**用于更新参数。**验证集**用于选择模型、学习率、轮数和阈值。**测试集**在这些决定完成后，用于最后一次独立评估。

如果每轮都根据测试集结果改模型，这部分数据事实上已经变成验证集，不能再把最终分数当成独立测试结果。

数据划分也要符合业务：用户重复出现时，要考虑按用户分组；用历史预测未来时，要考虑按时间划分。均值填充、标准化和类别词表等需要学习统计量的预处理，只能在训练集上拟合，再应用到其他集合。

“先对全量数据标准化，再切训练集”会让未来或评估数据的信息进入训练流程。这叫**数据泄漏**。模型有可能因此得到很好看的离线成绩，却无法在线复现。

### 7.3 一个完整且无需下载数据的训练示例

现在用 PyTorch 学习 $$y=2.5x+5.2+\text{噪声}$$。这个模型只有一个线性层，不算深度网络，但它把后面所有网络都要用到的训练动作展示完整了。

保存为 `linear_regression.py`，或直接运行[配套下载包](#download)中的 `examples/03_linear_regression.py`：

```python
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
torch.set_num_threads(2)
x = torch.randn(400, 1)
y = 2.5 * x + 5.2 + 0.1 * torch.randn(400, 1)
x_train, x_test = x[:320], x[320:]
y_train, y_test = y[:320], y[320:]
loader = DataLoader(
    TensorDataset(x_train, y_train), batch_size=32, shuffle=True
)
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
history = []

for epoch in range(80):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xb)
        loss = loss_fn(prediction, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    history.append(total_loss / len(x_train))

model.eval()
with torch.inference_mode():
    test_mse = loss_fn(model(x_test), y_test).item()
result = {
    "weight": model.weight.item(),
    "bias": model.bias.item(),
    "first_train_mse": history[0],
    "last_train_mse": history[-1],
    "test_mse": test_mse,
}
print(json.dumps(result, indent=2))
out = Path("runs/linear")
out.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), out / "model.pt")
(out / "metrics.json").write_text(json.dumps(result, indent=2))
(out / "history.json").write_text(json.dumps(history))

# Verify that persistence preserves predictions.
restored = nn.Linear(1, 1)
restored.load_state_dict(torch.load(out / "model.pt", weights_only=True))
restored.eval()
with torch.inference_mode():
    torch.testing.assert_close(restored(x_test), model(x_test))
```

![线性回归的拟合结果和实际训练损失](06-linear-training.png){: width="2038" height="796" }

*图 7：配套脚本的一次实际 CPU 运行。左图展示测试样本与学到的直线，右图展示训练过程中按样本加权汇总的 MSE。*

本次运行结果为：`w=2.4947，b=5.2017，测试 MSE=0.01254`。这些数字来自带噪声的合成数据，只用于确认模型能够学回预设关系。

训练中真正核心的动作只有几步：前向得到预测、计算损失、清空上一轮梯度、反向求导、优化器更新。代码把清空梯度放在前向之前，这样更容易避免漏清；放在上一次更新之后也可以，只要下一次反向前的状态正确。

注意损失汇总中的：

```text
累计损失 += 当前批次平均损失 × 当前批次样本数
整轮平均损失 = 累计损失 / 整轮样本总数
```

最后一批可能更小，直接把每批均值相加后除以样本数，会把指标算错。对本文这种每个样本输出维度相同、默认取均值的损失，按上述方式汇总即可。

### 7.4 eval 与关闭梯度是两个开关

`model.train()` 和 `model.eval()` 控制 Dropout、BatchNorm 等层的行为。

`torch.no_grad()` 和 `torch.inference_mode()` 控制是否记录求导信息；后者更适合这里纯推理的使用方式，但不适合所有后续还要参与自动微分的场景。

所以评估时通常需要同时切换模式并关闭梯度。单独 `eval()` 不会自动关闭梯度；单独 `no_grad()` 也不会自动关闭 Dropout。

<a id="optimization"></a>

## 8 模型学不动时 先检查优化过程

### 8.1 为什么梯度会消失或爆炸

反向传播需要连乘多个局部导数和权重相关的变换。在一个极度简化的标量例子中，连续乘 20 次 0.5，得到约 $$9.5\times10^{-7}$$；连续乘 20 次 1.5，则得到约 3325。

这说明了两种风险：

- **梯度消失**：靠前参数收到的梯度很小，学习缓慢。
- **梯度爆炸**：梯度过大，参数更新不稳定，甚至出现无穷值或 `NaN`。

真实网络使用矩阵，不能只看某一个权重是否大于 1 来判断，但这个连乘直觉很有用。

处理时先检查输入和目标尺度、激活函数、初始化、学习率与网络结构。必要时可使用梯度裁剪，它限制更新前梯度的幅度，但不能修复错误的数据或损失定义。带残差连接的结构也能改善深层网络的优化。

### 8.2 优化器是在改变更新策略

普通 SGD 只根据当前梯度决定本步更新。在狭长的损失曲面中，它可能在陡峭方向来回振荡，却沿缓慢方向前进得很慢。非凸问题还可能遇到平坦区域、鞍点和局部极小值；换优化器也不能保证找到全局最优。

| 方法 | 直觉 | 优点与局限 | PyTorch API |
|---|---|---|---|
| SGD | 沿当前负梯度前进 | 简单，但对学习率和曲面尺度敏感 | `optim.SGD` |
| Momentum | 给持续一致的方向积累“速度” | 有助于减轻振荡，也可能冲过头 | `optim.SGD(momentum=0.9)` |
| AdaGrad | 累积每个参数过去梯度的平方，缩小频繁更新方向的步幅 | 适合认识自适应步长，但累计量可能让后期步幅过小 | `optim.Adagrad` |
| RMSProp | 用衰减平均替代永久累计，逐渐忘掉很早的梯度 | 缓解 AdaGrad 的持续缩小问题，仍需要调参 | `optim.RMSprop` |
| Adam | 同时维护梯度均值与梯度平方的衰减平均，并做偏差修正 | 方便建立训练基线，但不保证效果最佳 | `optim.Adam` |

**自适应学习率不意味着不需要设置学习率。** 你传入的 `lr` 仍然决定整体更新尺度。

### 8.3 学习率衰减控制训练的节奏

初期使用相对大的步长，后期缩小步长，是一种常见策略：

| 策略 | 改变方式 | 对应类 |
|---|---|---|
| 等间隔衰减 | 每隔固定轮数乘一次 `gamma` | `StepLR` |
| 指定轮数衰减 | 到达指定里程碑时乘一次 `gamma` | `MultiStepLR` |
| 指数衰减 | 每次调度都乘一次 `gamma` | `ExponentialLR` |

若指数衰减的初始学习率是 $$\eta_0$$，经过 $$t$$ 次调度后是 $$\eta_0\gamma^t$$。不要每次再把“当前学习率”乘以 $$\gamma^t$$，那会衰减得更快。

![梯度尺度 优化轨迹和学习率调度](07-optimization.png){: width="2045" height="1424" }

*图 8：左上是标量连乘示意；右上是在指定二次函数上的实际数值迭代；下方是三种调度曲线。它们用于解释机制，不是优化器优劣排行榜。*

下面的例子可以独立运行。调度器在这一轮的参数更新完成后执行：

```python
import torch

w = torch.nn.Parameter(torch.tensor([3.0]))
optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
for epoch in range(6):
    optimizer.zero_grad(set_to_none=True)
    loss = w.square().mean()
    loss.backward()
    optimizer.step()
    print(epoch + 1, optimizer.param_groups[0]["lr"], loss.item())
    scheduler.step()
```

这里一次循环代表一轮。真实训练通常一轮包含多个 batch，所以应把 `scheduler.step()` 放在预先决定的调度位置。不是所有调度器都按 epoch 调用；使用前要确认它的语义。

优化器的公式集中在[附录 B](#optimizer-math)。初读时先掌握“更新用了哪些历史信息”即可。

<a id="initialization"></a>

## 9 让训练稳定 并让模型能够泛化

### 9.1 初始化不仅仅是随便生成一组小数

**初始化**决定模型从哪里开始学习。多层网络同一层的隐藏单元如果使用完全相同的权重与偏置，它们可能接收相同输入、得到相同梯度，一直学成相同功能。这叫**对称性问题**。

所以通常用随机初始化打破对称性。不过，“权重不能设为 0”也不是无条件定律：单个线性回归模型可以从零开始，偏置常常也可以初始化为零。需要警惕的是多神经元隐藏层的对称设置。

常用初始化如下：

| 方法 | 含义 | 适用提醒 |
|---|---|---|
| 常数初始化 | `zeros_`、`ones_`、`constant_` | 可以用于偏置等场景，隐藏层权重应注意对称性 |
| 单位矩阵初始化 | `eye_` | 对二维权重初始化出单位矩阵或矩形的单位映射部分；不叫“秩初始化” |
| 正态分布 | `normal_(mean, std)` | 尺度要合适，标准差不是越小越好 |
| 均匀分布 | `uniform_(a, b)` | 区间过大或过小都会影响信号传播 |
| Xavier / Glorot | 根据输入数和输出数控制尺度 | 常用于近似线性、Tanh 等场景，需考虑 gain |
| He / Kaiming | 根据输入数和激活函数控制尺度 | 常用于 ReLU 和 Leaky ReLU |

正态分布 $$\mathcal N(\mu,\sigma^2)$$ 表示均值为 $$\mu$$、方差为 $$\sigma^2$$ 的钟形分布；均匀分布 $$\mathcal U(a,b)$$ 表示区间内等长小区间具有相同概率。随机初始化关注的是尺度如何影响逐层传播，并不需要先背概率密度函数。

例如，一个使用 ReLU 的隐藏层可以这样初始化：

```python
from torch import nn

layer = nn.Linear(8, 16)
nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
nn.init.zeros_(layer.bias)
print(layer.weight.shape, layer.bias.shape)
```

`nn.Linear` 本身已经有默认初始化，不手动调用 `nn.init` 并不等于权重未初始化。Xavier 与 He 的尺度公式见[附录 C](#initialization-math)。

<a id="regularization"></a>

### 9.2 训练集表现好 并不等于学得好

模型记住训练样本的细节，却不能很好地处理新样本，这叫**过拟合**。能在新样本上保持效果，叫**泛化**。

训练与验证都表现较差，可能是欠拟合、优化失败、特征不足或标签问题；训练继续变好而验证变差，则应警惕过拟合。二者都不能仅凭单个 epoch 的波动下结论。

![训练和验证曲线如何帮助判断问题](08-generalization.png){: width="2039" height="850" }

*图 9：这是解释欠拟合与过拟合的示意曲线，并非某个数据集的实验结果。*

常用办法包括收集更有代表性的数据、减少模型复杂度、增加正则化，以及按照验证集结果停止训练。接下来三个工具解决的是不同问题，不宜一股脑全部加上。

<a id="batchnorm"></a>

### 9.3 BatchNorm 调整中间表示的尺度

Batch Normalization，简称 **BN**，在训练时对一组激活值进行标准化，再做可学习的缩放和平移：

$$
\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},
\qquad y=\gamma\hat x+\beta
$$

$$\mu_B$$、$$\sigma_B^2$$ 来自当前统计范围；$$\epsilon$$ 防止除以零；$$\gamma$$、$$\beta$$ 是可学习参数。对于 `[N,D]` 的 `BatchNorm1d(D)`，每个特征沿批次统计；`BatchNorm2d(C)` 对每个通道沿批次与空间位置统计。

把某个特征或通道参与统计的 $$m$$ 个值记为 $$x_1,\ldots,x_m$$，训练前向计算使用：

$$
\mu_B=\frac1m\sum_{i=1}^m x_i,\qquad
\sigma_B^2=\frac1m\sum_{i=1}^m(x_i-\mu_B)^2
$$

例如同一特征的值为 `[1,3,5,7]`，均值是 4，方差是 5；先减 4，再除以 $$\sqrt{5+\epsilon}$$，最后由 $$\gamma$$ 和 $$\beta$$ 调整模型需要的尺度与位置。

在默认配置下，训练时还会更新运行统计量，评估时使用已保存的运行统计量，而非对每个请求重新估计。注意两个方差的口径：当前批次前向使用上面的 $$1/m$$，更新 `running_var` 时使用 $$1/(m-1)$$ 的无偏估计。BN 的 `momentum` 控制运行统计量的更新比例，与优化器中的动量含义不同。[BatchNorm1d 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BatchNorm1d.html)

BN 的主要作用是改善优化时的数值行为，也可能带来正则化效果。不能把它当成防止过拟合的保证，更不能把它与输入数据的标准化混为一谈。小批次统计可能不稳定；在 `[N,D]` 的典型用法中，训练批次只有一条样本还可能直接报错。

`Linear → BatchNorm → ReLU` 是常见组合，但不是所有网络唯一正确的排列。

### 9.4 权重衰减限制参数规模

一种常见方法是在目标中增加 L2 惩罚：

$$
L_{\text{total}}=L_{\text{data}}+\frac{\lambda}{2}\sum_jw_j^2
$$

对权重求导后，多出 $$\lambda w$$。对于不带动量的普通 SGD，这相当于更新时先把权重乘上 $$1-\eta\lambda$$，再做梯度更新。

`weight_decay` 的具体语义取决于优化器。不能把所有优化器中的这个参数都简单等同为“在 loss 上加 L2”。例如 AdamW 把权重衰减与自适应梯度更新解耦；默认 Adam 的耦合 L2 方式与它不同。[AdamW 文档](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)

过拟合也不只是“权重太大”造成的。数据泄漏、样本太少、模型太复杂等问题，都不会因为添加一个 `weight_decay` 参数就自动消失。

<a id="dropout"></a>

### 9.5 Dropout 与早停

Dropout 在训练时以概率 $$p$$ 将部分激活置零，其余激活除以 $$1-p$$，使期望尺度保持不变。它让模型不容易只依赖某几个单元。评估时则保留所有单元，不再进行这种随机失活和缩放。

![BatchNorm 的统计变换和 Dropout 在训练评估阶段的差别](14-bn-dropout.png){: width="1978" height="1222" }

*图 10：上半部分沿同一特征的批次维度计算 BN；下半部分展示 `p=0.5` 的一次 Dropout 掩码。每个元素独立决定是否保留，不保证每次恰好丢弃一半。*

下面可以直接验证两种层在训练和评估时的不同。BN 在只观察一个批次后，运行统计量还未充分积累，评估输出与本批次训练输出不同是正常的。

```python
import torch
from torch import nn

x = torch.tensor([[1.], [3.], [5.], [7.]])
bn = nn.BatchNorm1d(1)
bn.train()
train_output = bn(x)
manual = (x - x.mean(0)) / torch.sqrt(x.var(0, correction=0) + bn.eps)
torch.testing.assert_close(train_output, manual)  # 默认 gamma=1、beta=0
bn.eval()
print("BN train / eval:", train_output.flatten().tolist(), bn(x).flatten().tolist())

torch.manual_seed(42)
dropout = nn.Dropout(0.5)
dropout.train()
print("Dropout train:", dropout(torch.ones(8)).tolist())  # 只可能为 0 或 2
dropout.eval()
torch.testing.assert_close(dropout(torch.ones(8)), torch.ones(8))
```

它可以被理解为训练许多共享参数的子网络所带来的一种近似集成效果，但不等于真正训练了多个独立模型。全连接层可使用 `nn.Dropout(p)`；卷积网络还应根据需要区分逐元素失活与按通道失活。

**早停 early stopping**根据验证集表现停止训练。例如验证损失连续若干轮没有改善，就停止，并恢复验证效果最好时的权重。保留最佳参数要用独立快照，例如 `copy.deepcopy(model.state_dict())`；只保存对当前张量的引用，后续训练可能继续改变它。

本文应用脚本运行固定轮数，然后恢复验证损失最低的一轮。它实现了“最佳模型选择”，没有把它冒充为提前结束训练的早停。

<a id="mnist"></a>

## 10 实战一 用全连接网络识别手写数字

### 10.1 把一张图变成一个分类任务

MNIST 中每张图片都是 28×28 的灰度图，对应 0～9 中的一个数字。输入可以先表示为 `[N,1,28,28]`，再展平为 `[N,784]`。

注意，MNIST 与从相同图像体系整理出来的竞赛 CSV 并不意味着训练测试划分完全相同，分数不应直接对照。这里使用 torchvision 提供的数据集入口，避免依赖某个没有随代码提供的本地文件。数据接口见 [MNIST 文档](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.MNIST.html)。

我们使用两个隐藏层：

```text
[N,1,28,28]
   → Flatten → [N,784]
   → Linear(784,50) → ReLU
   → Linear(50,100) → ReLU
   → Linear(100,10)
   → [N,10] logits
```

最后 10 个数是类别分数。分类可以直接取 `argmax(dim=1)`；若要显示概率，再调用 Softmax。Softmax 不改变最大元素所在的位置。

这个网络有：

$$
(784\times50+50)+(50\times100+100)+(100\times10+10)=45360
$$

个可学习参数。输入的 784 个像素本身不是模型参数。

### 10.2 运行完整训练与评估

完整脚本位于[配套下载包](#download)的 `examples/04_vision.py`，在解压后的目录中运行：

```bash
python examples/04_vision.py --task mnist --epochs 5 --limit 12000
```

首次运行会下载公开数据集，需要联网。默认从官方训练集随机取 12000 张，其中 9600 张训练、2400 张验证；官方 10000 张测试图片仅在训练与模型选择完成后评估。`--limit 0` 会使用完整训练池。

配套脚本为 MNIST 使用 HTTPS 数据镜像，为 Fashion-MNIST 使用发布方仓库中的文件，均保留 torchvision 的数据校验。若输入来自 CSV，则先读取标签列与 784 个像素列，再把像素转为浮点数、除以 255，并整理成 `[N,1,28,28]`；标签需要转为 `long`。不要把标签列一起当作像素。

脚本依次完成：像素转换为 `[0,1]` 的 `float32`、数据划分、DataLoader 分批、前向、交叉熵、反向更新、每轮验证、保存最佳权重、独立测试，以及重新加载后比较输出。

**像素除以 255**使用的是预先已知的编码范围，不需要估计全量数据统计量；它与“用所有样本拟合 MinMaxScaler”不是一回事。

训练循环已经在第 7 节完整展开，这里变化的主要是模型和数据。完整脚本保留了这些动作，读者可以直接对照两个文件。

本次 CPU 运行得到：`测试准确率 93.08%，测试交叉熵 0.2423`。这是小规模教学实验，不代表完整数据训练的最佳成绩，也不代表识别真实业务图片的效果。

### 10.3 保存参数后怎样使用

脚本会产生 `runs/mnist/model.pt`。它保存的是 `state_dict`，其中通常包含参数和注册的缓冲区，例如 BN 的运行统计量；它不自动保存数据集、类别含义和任意 Python 结构。

下面是与该模型结构匹配的完整加载示例。先运行上面的训练命令，再把这段保存为脚本执行：

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 50), nn.ReLU(),
    nn.Linear(50, 100), nn.ReLU(),
    nn.Linear(100, 10),
)
state = torch.load("runs/mnist/model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()
with torch.inference_mode():
    # 仅示范输入协议；全零图片不代表真实业务样本。
    sample = torch.zeros(1, 1, 28, 28)
    probabilities = model(sample).softmax(dim=1)
print(probabilities.shape)
```

部署要加载可信的模型文件，并同时保存类别映射、输入尺寸、像素缩放方式和模型结构版本。若要中断后继续训练，还需要优化器、调度器以及训练进度等状态。

<a id="housing"></a>

## 11 实战二 房价回归让我们看见特征工程

图像已经有规则的像素网格，业务表格却经常混合数字、字符串和缺失值。房价回归是一个很适合说明这些问题的例子。

假设输入是面积、房龄和区域，标签是价格。这里存在三类工作：

1. 数值特征先填补缺失值，再按训练集统计量标准化。
2. 类别特征先处理缺失，再做独热编码。独热编码用一组 0/1 列表示类别，避免随意赋予区域编号一个大小关系。
3. 把处理后的特征组合成固定列顺序的浮点矩阵，标签整理成 `[N,1]`。

这里的**标准化**通常指逐列做 $$x'=(x-\mu_{\text{train}})/s_{\text{train}}$$，其中均值和标准差只从训练数据计算。它让面积、房龄等量纲不同的字段处于较接近的数值尺度，不会把任意分布变成正态分布，也不会强制所有值都落在 `[0,1]`。`StandardScaler` 对常数列使用缩放因子 1，避免除以零。[StandardScaler 文档](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.StandardScaler.html)

`Pipeline` 让处理步骤顺序固定，`ColumnTransformer` 让不同列采用不同处理方式。它们的重要价值是让训练和推理复用同一套已拟合的转换。

下面这个预处理示例可以独立运行：

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

train = pd.DataFrame({
    "area": [60.0, 90.0, 120.0],
    "age": [10.0, np.nan, 2.0],
    "district": ["A", "B", "A"],
})
future = pd.DataFrame({"area": [80.0], "age": [np.nan], "district": ["C"]})
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("fill", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ]), ["area", "age"]),
    ("category", Pipeline([
        ("fill", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), ["district"]),
])
print(preprocessor.fit_transform(train))
print(preprocessor.transform(future))
print(preprocessor.get_feature_names_out())
```

新出现的区域 C 不会导致编码器报错，这里会得到该类别组的全零编码。但这不意味着模型“理解了 C”；真实系统需要记录未知类别比例。唯一标识 `Id` 等字段也不应仅因为是数字就自动送入模型，要判断它是否提供了在推理时稳定可用的信息。

对于小型教学数据，上面显式使用稠密输出。高基数类别可能让独热矩阵非常大，不能无条件 `.toarray()`；同时也不能假设 `ColumnTransformer` 的结果永远是稀疏矩阵。相关输入约定见 [OneHotEncoder 文档](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)。

### 11.1 为什么价格常用对数尺度

同样是误差 10 万元，对 20 万与 1000 万的目标，其相对影响差别很大。若我们希望更关注比例偏差，可以考虑：

$$
\operatorname{logRMSE}
=\sqrt{\frac1N\sum_i(\log\hat y_i-\log y_i)^2}
$$

当相对误差较小时，$$\log\hat y-\log y$$ 近似相对误差。但 logRMSE 并不等于平均百分比误差。

直接让网络输出任意价格，再把小于 1 的预测强行裁剪为 1 后取对数，会让被裁剪区域的梯度变成 0。一个更直接的办法是：

**让模型预测对数价格，在对数空间训练，再用指数函数还原。**

这要求价格严格为正。如果目标可能为零，应根据业务定义考虑 `log1p` 与 `expm1` 等变换，而不是把不同指标混在一起。

### 11.2 运行一个不依赖竞赛文件的回归实验

运行[配套下载包](#download)中的 `examples/05_house_prices.py`：

```bash
python examples/05_house_prices.py
```

它会自行生成 1200 条合成数据，划分为 720 条训练、240 条验证、240 条测试。数据带有类别变量和缺失值；预处理仅在训练集拟合。

模型结构是：

```text
预处理后的特征
→ Linear → BatchNorm → ReLU → Dropout → Linear
→ 标准化后的对数价格
```

训练使用 MSE、AdamW 和验证集最佳权重选择。标签的均值与标准差也仅从训练集计算。最后先还原对数尺度，再计算整个测试集的 logRMSE，并与“始终预测训练集平均对数价格”的常数基线比较。

具体来说，训练目标是 $$t=(\log y-\mu_{\log y})/s_{\log y}$$；网络输出 $$\hat t$$ 后，先得到 $$\widehat{\log y}=\hat t\,s_{\log y}+\mu_{\log y}$$，再用 $$\hat y=\exp(\widehat{\log y})$$ 还原价格。评估 logRMSE 时使用还原后的对数值，不能直接把标准化空间的 RMSE 当成原指标。

这个脚本完成了训练与评估。若要部署，还需要一起保存已拟合的预处理器、特征列顺序、目标变换统计量和模型权重；仅保存某一层参数无法复现完整预测。

本次结果为：`测试 logRMSE=0.09775，常数基线 logRMSE=0.41346`。数据是人为生成的，**这些分数不能用于判断任何真实地区的房价预测能力**。

如果换成真实业务数据，最先改的往往不是网络层数，而是按时间或实体划分数据、核对字段在预测时是否已经可得、处理异常和未知类别，再比较简单基线。对数空间拟合后直接取指数，也不自动保证得到原尺度上的条件均值，偏差是否需要修正取决于业务目标。

<a id="cnn"></a>

## 12 CNN 为什么更适合处理图像

### 12.1 全连接网络没有主动利用局部结构

把一张图片展平并不会删除像素值，但模型结构不再显式表达“这两个像素在空间上相邻”。对于大图片，全连接层的参数量也会很快变大。

**卷积神经网络 CNN**引入两个很有用的约束：

- **局部连接**：每个输出位置只观察输入的一小块区域。
- **权重共享**：同一组检测规则在不同空间位置重复使用。

一个用于识别局部边缘的规则，不必在图片左上角、右下角分别学习一遍。这个偏好叫一种**归纳偏置**：模型结构提前表达了我们认为数据可能具有的规律。

**感受野 receptive field**指一个输出位置在原始输入上能够受到影响的区域。第一层 3×3 卷积看一个 3×3 邻域；连续两层 3×3 卷积，在步幅和膨胀率均为 1 时，内部位置的理论感受野扩大到 5×5。这解释了局部计算如何逐层整合更大范围的信息。

CNN 常用于图像分类、目标检测和分割，也可以处理其他具有局部结构的数据。卷积层负责提取局部表示；激活函数引入非线性；池化或带步幅的卷积缩小空间尺寸；最后再整合特征得到输出。

### 12.2 一个卷积核怎样计算

**卷积核 kernel**可以先理解成一小块可学习的权重。把它放在输入某个位置，将覆盖区域与权重逐元素相乘，再求和、加偏置，然后滑到下一个位置。

![卷积窗口的逐元素相乘与求和](09-convolution.png){: width="1978" height="999" }

*图 11：3×3 的窗口在 4×4 输入上移动，步幅为 1、无填充，得到 2×2 输出。第一格按图中高亮区域计算。*

深度学习框架中通常把这种不翻转卷积核的互相关运算也称为卷积。卷积核是训练得到的权重；手工边缘核可以帮助理解，但随机初始化的卷积输出不代表已经学会提取边缘。

**特征图 feature map**就是一层卷积输出的空间响应。它可以有负值，也可以有很多通道，不一定能当成正常 RGB 图片直接显示。

<a id="padding-stride"></a>

### 12.3 Padding 与 stride 决定输出大小

**Padding，填充**，是在输入边缘补值，例如补零。它可以控制输出尺寸，并让边界位置有合适的参与方式。

**Stride，步幅**，是窗口每次移动的距离。增大步幅通常会缩小空间尺寸。

对高度方向，卷积核大小为 $$K$$、填充为 $$P$$、步幅为 $$S$$、膨胀率为 $$D$$ 时：

$$
H_{\text{out}}
=\left\lfloor
\frac{H_{\text{in}}+2P-D(K-1)-1}{S}+1
\right\rfloor
$$

本文使用普通卷积，即 $$D=1$$，公式简化为：

$$
H_{\text{out}}
=\left\lfloor\frac{H_{\text{in}}+2P-K}{S}\right\rfloor+1
$$

宽度同理。28×28 输入、3×3 卷积核、步幅 1、填充 1，输出仍是 28×28；填充改为 0，输出就是 26×26。卷积保留空间维度的组织方式，不代表长宽一定不变。[Conv2d 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html)

![相同输入在不同填充和步幅下的窗口与输出大小](15-padding-stride.png){: width="1815" height="882" }

*图 12：固定 4×4 输入和 3×3 核，比较 `(P,S)=(0,1)、(1,1)、(1,2)`。蓝色和橙色框是水平方向相邻的两个窗口；第三组的步幅为 2，因此每次移动两格。*

公式中的向下取整不能省略。例如最后一种情况，$$(4+2-3)/2+1=2.5$$，输出大小取 2，不会创建半个输出位置。

<a id="multichannel"></a>

### 12.4 多通道卷积怎样工作

普通 `groups=1` 的卷积中，一个输出通道对应一个覆盖**所有输入通道**的卷积核。每个输入通道分别与对应权重做乘加，再把各通道的结果求和，并加一个偏置。

RGB 图片具有通道、高、宽三个轴，并不意味着应该使用 `Conv3d`。这里的 `Conv2d` 在高、宽两个空间维度上滑动，同时处理输入通道。

因此，`Conv2d(3,16,kernel_size=3)` 的权重形状为 `[16,3,3,3]`。它有 16 组核，每组核接收 3 个输入通道，产生 1 个输出通道。

带偏置时参数量为：

$$
16\times3\times3\times3+16=448
$$

参数量不直接乘输入图片的高宽，计算量却会随输出空间大小增加。把“参数少”理解成“任何情况下都计算便宜”，也会误判性能。

![多个输入通道先分别卷积 再相加得到一个输出通道](16-multichannel.png){: width="1978" height="1124" }

*图 13：以两个输入通道、两个输出通道为例。每个输出通道都有独立的一组跨通道权重和一个偏置；通道内乘加的结果需要求和。*

用最简单的 1×1 核核算一次：输入同一位置的两个通道是 `[2,5]`，第一组权重为 `[1,2]`、偏置为 1，输出是 `2×1+5×2+1=13`；另一组权重为 `[-1,1]`、偏置为 0，输出是 3。输出通道数由权重组数决定。

```python
import torch
from torch import nn

x = torch.tensor([[[[2.]], [[5.]]]])  # [N=1,C=2,H=1,W=1]
conv = nn.Conv2d(2, 2, kernel_size=1)
with torch.no_grad():
    conv.weight.copy_(torch.tensor([[[[1.]], [[2.]]], [[[-1.]], [[1.]]]]))
    conv.bias.copy_(torch.tensor([1., 0.]))
y = conv(x)
print(y.flatten().tolist())  # [13.0, 3.0]
torch.testing.assert_close(y.flatten(), torch.tensor([13., 3.]))
```

因此，**1×1 卷积也有实际作用**：它在每个空间位置混合通道，并能改变通道数量；在步幅为 1 时，它本身不扩大感受野。后面的 Inception 会用它先压缩通道，减少较大卷积核的计算开销。

### 12.5 池化如何压缩空间信息

**最大池化 Max Pooling**取局部窗口里的最大值；**平均池化 Average Pooling**取均值。普通池化按通道分别处理，没有需要训练的权重，通常不改变通道数。

![同一输入经过最大池化和平均池化](10-pooling.png){: width="1978" height="968" }

*图 14：2×2 窗口、步幅 2，把 4×4 压缩为 2×2。最大池化保留强响应，平均池化保留局部平均水平。*

池化可以减少后续计算，也可能降低对局部微小位置变化的敏感程度，但它会丢失信息，并不保证对任意平移都不变，更不是防止过拟合的万能方法。检测很小的目标、定位精细边界时，过早下采样可能损害效果。

下面的例子可独立运行：

```python
import torch
from torch import nn

x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 1, 4, 4)
print(nn.MaxPool2d(2, stride=2)(x))
print(nn.AvgPool2d(2, stride=2)(x))

conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
print(conv(x).shape)  # [1,8,4,4]
print(sum(p.numel() for p in conv.parameters()))  # 8*(1*3*3)+8 = 80
```

<a id="fashion"></a>

### 12.6 实战三 用 CNN 完成服装分类

Fashion-MNIST 同样是 28×28 灰度图，但类别变成服装和配饰。标签 0～9 依次对应：T 恤或上衣、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、踝靴。[Fashion-MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist)

本文使用一个小型 CNN：

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 7 * 7, 10),
)
x = torch.randn(2, 1, 28, 28)
for layer in model:
    x = layer(x)
    print(type(layer).__name__, tuple(x.shape))
print("parameters:", sum(p.numel() for p in model.parameters()))  # 9098
```

这段代码只检查计算和形状，随机权重还没有识别能力。完整训练继续使用[配套下载包](#download)中的 `examples/04_vision.py`：

```bash
python examples/04_vision.py --task fashion --epochs 5 --limit 12000
```

![小型 CNN 的数据形状变化](11-cnn-shapes.png){: width="2138" height="905" }

*图 15：每个卷积块保留或增加通道，再通过池化减小长宽。展平前的 `[N,16,7,7]` 决定了线性层需要 784 个输入。*

数据划分、损失函数和训练循环与数字分类相同，变化的是数据内容与模型结构。本次 CPU 运行得到：`测试准确率 83.87%，测试交叉熵 0.4551`。

![数字与服装分类的实际预测样本](12-predictions.png){: width="2039" height="1055" }

*图 16：从各自官方测试集取出的样本，使用配套脚本保存并重新加载的最佳模型预测。展示固定位置样本，不按正确与否挑选。数据说明见[图片说明](#image-credits)。*

如果读到 LeNet 风格的网络，会看到另一种常见结构：5×5 卷积、Sigmoid、平均池化，以及多层全连接。理解形状后，两者并不难对照：

```text
[N,1,28,28]
→ Conv(1,6,5,padding=2) → [N,6,28,28]
→ AvgPool(2)           → [N,6,14,14]
→ Conv(6,16,5)         → [N,16,10,10]
→ AvgPool(2)           → [N,16,5,5]
→ Flatten             → [N,400]
→ Linear(400,120) → Linear(120,84) → Linear(84,10)
```

上面只列出改变形状或带参数的步骤。完整的这个 LeNet 风格变体在两个卷积层、前两个全连接层之后放置 Sigmoid，最后一层输出 logits。采用 Sigmoid 时可考虑与之匹配的 Xavier 初始化；采用 ReLU 时则要考虑 He 初始化。本文实际训练的 CNN 使用 ReLU 和最大池化，以便在小规模练习中直接观察训练过程。

<a id="architectures"></a>

## 13 从经典 CNN 中认识几种结构思想

不必从一开始背完所有网络。先看它们分别改变了什么：

| 网络 | 关键思想 | 对初学者的启发 |
|---|---|---|
| AlexNet | 使用较深的 CNN，结合 ReLU、GPU 训练与 Dropout；有 5 个卷积层和 3 个全连接层 | 模型、数据、优化和计算资源要一起考虑 |
| VGG | 反复堆叠小尺寸的 3×3 卷积，形成 VGG-16、VGG-19 等结构 | 简单模块也可以组成很深的网络，但参数和计算成本仍然需要评估 |
| GoogLeNet / Inception | 并行使用不同尺度的分支，再沿通道拼接，并用 1×1 卷积等控制开销 | 网络除了加深，也可以组合不同尺度的信息 |
| ResNet | 通过残差连接，让模块学习 $$F(x)$$，输出 $$x+F(x)$$ | 为信息和梯度提供较直接的路径，改善深层网络的优化 |

这些设计分别可参阅原论文：[AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)、[VGG](https://arxiv.org/abs/1409.1556)、[GoogLeNet](https://arxiv.org/abs/1409.4842)、[ResNet](https://arxiv.org/abs/1512.03385)。

VGG-16、VGG-19 中的数字计入卷积层和全连接层，不计入池化、ReLU 等无权重操作。比较“多少层”时，要先统一计数方式。

![Inception 沿通道拼接分支 残差连接逐元素相加](17-inception-residual.png){: width="2218" height="1315" }

*图 17：左侧为带通道压缩的 Inception 结构示意，右侧为残差分支示意。省略部分激活与规范化层，重点看分支怎样汇合。*

Inception 的各分支需要保持相同的空间大小，才能沿通道维拼接。例如两个输出分别是 `[N,16,H,W]` 和 `[N,32,H,W]`，拼接后得到 `[N,48,H,W]`。残差连接则是对应位置相加，两侧形状匹配时，输出形状保持不变。二者不是同一种“合并”。图中的分支和通道压缩方式可对照 [GoogLeNet 论文](https://arxiv.org/abs/1409.4842)。

如果目标映射是 $$H(x)$$，残差模块可以学习 $$F(x)=H(x)-x$$，最终输出 $$x+F(x)$$。当输入输出形状不同时，需要投影等方式使形状匹配。它改善了深层网络难训练、加深后退化的问题，但不保证任何配置都不会出现梯度消失或爆炸。

ImageNet 的 ILSVRC 分类任务涉及 1000 个类别，2012 年前后 CNN 在这类视觉任务上的进展推动了深度学习的发展。认识这段背景，是为了理解为什么研究开始关注更深的结构、训练方法和算力，而不是把竞赛排名当成业务选型结论。

“更深”也不自动意味着“参数更少”。例如在输入输出通道都为 $$C$$、不考虑偏置时，两层 3×3 卷积的参数量是 $$18C^2$$，小于一层 5×5 的 $$25C^2$$，且在步幅为 1 时具有相同的理论感受野大小；这属于特定结构替换带来的收益，不能推广到任意加层操作。

<a id="engineering"></a>

## 14 把模型放进业务系统之前 还要想清楚什么

### 14.1 模型交付物不只有一个权重文件

一个预测函数至少还依赖输入字段顺序、缺失值规则、类别映射、归一化方式和模型结构。将这些信息与权重绑定版本，才能避免“模型没变，特征变了”的问题。

例如图片服务训练时使用灰度、28×28、像素除以 255，线上却读成 RGB、直接使用 0～255 数值，模型即使正常返回 200，也不代表预测有效。

后端接入时，一条实用链路是：

```text
请求校验
→ 按训练约定预处理
→ 转成正确 shape / dtype / device 的批次
→ 在 eval 与关闭梯度的状态下推理
→ 将结果还原为业务输出
→ 记录模型版本、输入异常、延迟和后续真实结果
```

服务通常在进程启动时加载模型，避免每个请求重复读文件。实际还要处理超时、并发、批处理、资源限制、回退策略，以及版本回滚。

### 14.2 分数高不代表可以直接自动决策

准确率适合观察本文相对均衡的十分类练习，但类别极不均衡时可能失真。假如 99% 的样本属于正常类，始终预测正常也有 99% 准确率，却可能完全没有解决异常识别问题。

此时需要结合精确率、召回率、混淆矩阵和误判成本。把异常定义为正类，**精确率 precision**回答“预测为异常的样本里，多少真的异常”，**召回率 recall**回答“所有真实异常里，找出了多少”。例如标记了 20 条异常，其中 15 条是真的，而数据中实际有 30 条异常，则精确率为 `15/20=75%`，召回率为 `15/30=50%`。

**混淆矩阵**按真实类别和预测类别交叉计数，能看出模型把哪些类别混在一起；使用图表时先确认行、列各表示什么。分类阈值在验证集上选择；测试集负责独立评估。

Softmax 的最大值也不等于“这次预测一定正确的可信度”。模型可能在从未见过的输入上非常自信。**概率校准**关注“预测概率约为 80% 的一组样本，是否约有 80% 真的正确”；**拒识**则允许模型把不适合自动处理的输入交给回退流程。业务上还要观察分布变化和人工复核策略。

### 14.3 常见问题可以按症状排查

| 症状 | 优先检查 |
|---|---|
| 第一层就报形状错误 | batch 维是否缺失，图片是否为 NCHW，Linear 输入维度是否匹配 |
| loss 没报错但效果很怪 | 标签是否错位，回归是否意外广播，训练和推理预处理是否一致 |
| 分类 loss 不下降 | 标签类型与范围，是否给 CrossEntropy 重复加 Softmax，学习率，梯度是否被 detach |
| 参数似乎没更新 | 是否调用 backward 与 step，优化器是否拿到了对应参数，是否误用 no_grad |
| 后几轮突然出现 NaN | 输入与标签是否有 NaN/Inf，学习率是否过大，取对数的输入是否有效 |
| 同一输入每次预测不同 | 是否漏掉 eval，是否仍有随机数据增强或其他随机逻辑 |
| 验证很好 上线很差 | 数据泄漏、时间分布变化、实体重复、特征可得性和预处理版本 |
| 最后一批触发 BN 错误 | 批次是否只有一条有效统计样本，是否该调整 batch 或规范化方案 |
| 显存持续增长 | 是否把带计算图的 loss 或输出存入列表；日志通常用 item 或 detach |

固定随机种子有助于复现，但不保证不同 PyTorch 版本、设备和算子实现下逐位一致。因此实验记录还要包含环境、数据版本、划分方式和训练配置。

<a id="next"></a>

## 15 下一步怎么学才不会只会照抄代码

先做三个小练习，每次只改一个因素：

1. 在线性回归中把学习率分别调大和调小，记录损失曲线，解释为什么会慢或者振荡。
2. 在数字分类中删除一个隐藏层，比较参数量、验证损失与最终测试效果。不要只比较训练准确率。
3. 在服装分类中替换池化方式，先手算每层形状，再运行代码验证。固定数据划分，并把选择依据留在验证集。

之后可以按任务继续：

- 对基础概念仍有疑问，继续读 [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)，对照自己的训练脚本。
- 想补齐数学与实现之间的联系，可阅读 [动手学深度学习](https://zh.d2l.ai/)，优先看线性网络、多层感知机、优化与卷积章节。
- 做真实图像业务时，下一步学习迁移学习、数据增强和预训练模型微调，比较从零训练与复用已有表示的成本。
- 想进入大语言模型方向，再学习词元化、Embedding、注意力、Transformer 和微调。先弄清“输入怎样表示、输出优化什么、数据怎样评估”，会更容易读懂这些模型。

开始真正的业务实验时，可以先写下四个答案：预测发生在什么时候，模型当时能拿到哪些字段，什么结果算预测正确，以及误判会带来什么成本。把这四件事说清楚，再写训练代码，模型才有机会解决一个明确的问题。

<a id="tensor-api"></a>

## 附录 A PyTorch 张量操作速查

这些 API 不必背诵。遇到需求时，根据“创建、类型、计算、选择、形状”查找即可。以下例子默认已经 `import torch`。

### A.1 创建与类型转换

| 目的 | API 或示例 | 需要记住的区别 |
|---|---|---|
| 按数据创建 | `torch.tensor([[1,2],[3,4]], dtype=torch.float32)` | 明确数据和类型 |
| 按形状预留空间 | `torch.empty(2,3)` | 未初始化，不能直接当有效输入 |
| 旧式创建 | `torch.Tensor(2,3)`、`torch.FloatTensor(...)` 等 | 同一入口可能因参数形式不同而有不同含义，优先使用显式工厂函数 |
| 整数类型 | `int16` / `ShortTensor`，`int32` / `IntTensor`，`int64` / `LongTensor` | 转整数会丢失小数部分 |
| 浮点类型 | `float32` / `FloatTensor`，`float64` / `DoubleTensor` | 本文主要使用 float32 |
| 按步长生成 | `torch.arange(0,6,2)` | 右端点不包含 |
| 按点数生成 | `torch.linspace(0,1,5)` | 包含两端点 |
| 指数间隔 | `torch.logspace(1,3,3,base=2)` | 指数从 1 到 3，结果为 2、4、8 |
| 固定值填充 | `zeros`、`ones`、`full` | `torch.full((2,3), 7.0)` |
| 依照已有张量 | `zeros_like`、`ones_like`、`full_like`、`empty_like` | 复用形状等属性；empty_like 仍未初始化 |
| 单位矩阵 | `torch.eye(3)` 或 `torch.eye(3,4)` | 主对角线为 1 |
| 均匀随机数 | `rand`、`rand_like` | `[0,1)` |
| 随机整数 | `randint`、`randint_like` | 下界包含、上界不包含；like 版本默认沿用输入 dtype |
| 标准正态分布 | `randn`、`randn_like` | 均值 0、方差 1 |
| 指定正态分布 | `torch.normal(mean=2., std=0.5, size=(2,3))` | `std` 是标准差 |
| 随机排列 | `torch.randperm(n)` | 把 `0…n-1` 打乱，不是有放回采样 |
| 随机种子 | `manual_seed(42)`、`torch.random.initial_seed()` | 设置种子与查看初始种子；不能据此保证跨平台完全相同 |
| 转类型 | `.to(dtype=...)`、`.type(...)`、`.float()`、`.double()`、`.long()` | 接住返回值 |
| 与 NumPy 互转 | `from_numpy`、`tensor`、`.numpy()` | 注意设备、梯度状态和内存共享 |
| 提取 Python 标量 | `.item()` | 只适用于恰好一个元素的张量 |

### A.2 计算与统计

| 目的 | API | 提醒 |
|---|---|---|
| 四则运算 | `+ - * /`，`add/sub/mul/div` | `*` 是逐元素乘法 |
| 取负与幂 | `neg`、`**`、`pow` | 保持与数学定义一致的输入 |
| 开方 指数 对数 | `sqrt`、`exp`、`log` | 负数实数开方、非正数实数对数等会产生非有限结果 |
| 原地修改 | `add_`、`sub_`、`mul_`、`div_`、`neg_`、`pow_`、`sqrt_`、`exp_`、`log_` | 留意 dtype 与自动求导约束 |
| 二维矩阵乘法 | `torch.mm(a,b)` | 仅二维，不自动批次广播 |
| 通用矩阵乘法 | `a @ b`、`torch.matmul(a,b)` | 高维时最后两维做矩阵乘法，前面的批次维可广播；一维输入有向量规则 |
| 求和与均值 | `sum`、`mean` | 用 `dim` 指定被聚合的轴；`keepdim=True` 保留长度为 1 的轴 |
| 极值 | `max`、`min` | 指定单个 dim 时返回值与索引；全量归约则返回一个值 |
| 极值索引 | `argmax`、`argmin` | 分类时通常沿类别维操作 |
| 标准差 | `std` | PyTorch 2.9 默认 `correction=1`；总体标准差可用 `correction=0` |
| 去重与排序 | `unique`、`sort` | sort 同时返回排序值和原索引 |

### A.3 选择与形状

| 需求 | 示例 | 含义 |
|---|---|---|
| 简单索引 | `x[0]`、`x[1,2]` | 选择指定位置，部分轴可能消失 |
| 范围索引 | `x[:,1:4]` | 左闭右开 |
| 成对索引 | `x[[0,1],[1,2]]` | 选 `(0,1)` 与 `(1,2)` |
| 交叉选择 | `x[[[0],[1]],[1,2]]` | 利用索引广播得到两行与两列的组合 |
| 布尔索引 | `x[x>0]` | 掩码形状要与被索引的维度对应 |
| 交换两个轴 | `x.transpose(0,1)` | 交换轴，不是重新填数据 |
| 重排多个轴 | `x.permute(0,3,1,2)` | 例如 NHWC 转 NCHW |
| 调整形状 | `reshape`、`view` | 元素总数相同；`-1` 推断一个维度 |
| 检查连续性 | `is_contiguous`、`contiguous` | 与内存布局有关 |
| 增加长度为 1 的轴 | `unsqueeze(dim)` | 单张图可以补上 batch 维 |
| 删除长度为 1 的轴 | `squeeze(dim)` | 尽量明确要删除哪个轴 |
| 沿原轴拼接 | `torch.cat([a,b],dim=0)` | 其他维度大小须匹配 |
| 沿新轴堆叠 | `torch.stack([a,b],dim=0)` | 各输入形状须一致 |

<a id="optimizer-math"></a>

## 附录 B 把优化器公式与代码名字对应起来

下面用 $$\theta_t$$ 表示第 $$t$$ 步参数，用 $$g_t$$ 表示当前梯度。平方、开方和除法都按元素进行；$$\epsilon$$ 是避免分母为零的小常数。为便于比较，省略权重衰减及各类可选变体。

**SGD**：

$$
\theta_t=\theta_{t-1}-\eta g_t
$$

**Momentum**的一种常见记法：

$$
v_t=\mu v_{t-1}+g_t,\qquad
\theta_t=\theta_{t-1}-\eta v_t
$$

历史方向通过 $$\mu$$ 衰减后参与下一步。学习率恒定且初值对应时，可以把学习率和负号并入速度变量，得到等价记法；若学习率随步骤变化，直接把它移入历史累积式，就不一定保持等价。上式与 PyTorch 在未启用 dampening、Nesterov 等变体时的常用更新形式对应。[SGD 文档](https://docs.pytorch.org/docs/2.9/generated/torch.optim.SGD.html)

**AdaGrad**：

$$
s_t=s_{t-1}+g_t^2,\qquad
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}
$$

累计平方梯度较大的方向，后续有效步幅更小。累计量并非任何情况下都无限增长，因此“最终一定停止更新”也不是普适结论。

**RMSProp**：

$$
s_t=\alpha s_{t-1}+(1-\alpha)g_t^2,\qquad
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{s_t}+\epsilon}
$$

它不永久等权保存所有历史，而是对更早的梯度逐步降低权重。

**Adam**：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

当初始状态为零时，早期的衰减平均存在偏向零的初始化偏差，因此使用：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t=\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

这里的一阶矩用于估计方向，二阶原始矩用于估计尺度。Adam 更接近将动量思想与 RMSProp 式的衰减平方梯度结合，再加入偏差修正；不能把它理解成把两个优化器依次执行一遍。

<a id="derivatives"></a>

## 附录 C 关键梯度和初始化公式

### C.1 激活函数的局部梯度

| 函数 | 导数或梯度规则 |
|---|---|
| Identity | $$1$$ |
| 阶跃 | 非阈值位置为 0，阈值处不可导 |
| Sigmoid | $$\sigma(z)(1-\sigma(z))$$，最大值是 $$0.25$$ |
| Tanh | $$1-\tanh^2(z)$$ |
| ReLU | 正半轴为 1，负半轴为 0；0 点采用约定值 |
| Leaky ReLU / PReLU | 正半轴为 1，负半轴为斜率 $$\alpha$$；PReLU 还要学习 $$\alpha$$ |
| RReLU | 与本次采用的负半轴斜率对应 |
| ELU | 正半轴为 1，负半轴为 $$\alpha e^z$$ |
| SiLU | $$\sigma(z)+z\sigma(z)(1-\sigma(z))$$ |
| Softplus | $$\sigma(z)$$，这里指默认数学形式 |

已知上游梯度后，再乘局部导数即可。例如 ReLU 会把负输入位置的梯度截成 0，Sigmoid 会乘上它输出值与 `1-输出值` 的乘积。

![阶跃 恒等 负半轴斜率 ELU SiLU Softplus 的曲线对照](18-activation-variants.png){: width="1832" height="1242" }

*图 18：补充激活函数的形状。PReLU 的斜率通过训练学习，RReLU 的斜率在训练时随机采样；图中只展示可能的斜率族，不表示它们参数固定。*

### C.2 全连接层的梯度也可以检查形状

采用 $$Y=XW+b$$ 的数学约定：

- $$X$$ 为 `[N,D]`；
- $$W$$ 为 `[D,H]`；
- $$Y$$ 为 `[N,H]`；
- 上游梯度 $$G=\partial L/\partial Y$$ 为 `[N,H]`。

则：

$$
\frac{\partial L}{\partial X}=GW^\top,\qquad
\frac{\partial L}{\partial W}=X^\top G,\qquad
\frac{\partial L}{\partial b}=\sum_{i=1}^{N}G_{i,:}
$$

梯度形状分别与 $$X$$、$$W$$、$$b$$ 一致。偏置被广播给整个 batch，因此反向时需要把各样本对它的贡献相加。

### C.3 Softmax 与交叉熵为什么搭配自然

令 $$p=\operatorname{softmax}(z)$$，则：

$$
\frac{\partial p_k}{\partial z_j}
=p_k(\delta_{kj}-p_j)
$$

$$\delta_{kj}$$ 在 $$k=j$$ 时为 1，否则为 0。Softmax 的一个输出依赖所有输入分数，所以不能把它当成逐元素的 Sigmoid。

对于单个样本和归一化标签分布 $$y$$：

$$
L=-\sum_k y_k\log p_k
=-\sum_k y_kz_k+\log\sum_j e^{z_j}
$$

对 $$z_j$$ 求导，得到：

$$
\frac{\partial L}{\partial z_j}=p_j-y_j
$$

若对 $$N$$ 个样本取平均，梯度再除以 $$N$$。这里假设没有类别权重、忽略标签等额外设置。

一个直观例子：正确类别的预测概率只有 0.2，它对应的梯度是 `0.2-1=-0.8`，梯度下降会推动该类别的分数增大。其他类别的标签为 0，其梯度为正，更新会推动分数减小。

<a id="initialization-math"></a>

### C.4 Xavier 与 He 怎样控制尺度

设 `fan_in` 为输入连接数，`fan_out` 为输出连接数。在最简单的 gain 为 1 的 Xavier 形式中：

$$
W_{\text{Xavier normal}}
\sim\mathcal N\left(0,\frac{2}{\text{fan\_in}+\text{fan\_out}}\right)
$$

$$
W_{\text{Xavier uniform}}
\sim\mathcal U\left(
-\sqrt{\frac6{\text{fan\_in}+\text{fan\_out}}},
\sqrt{\frac6{\text{fan\_in}+\text{fan\_out}}}
\right)
$$

正态分布第二个参数是方差，因此传给 `normal_` 的标准差要开平方。使用 Tanh 等激活时，还可结合对应的 gain 调整尺度。

对于 ReLU、`fan_in` 模式的 He 初始化：

$$
W_{\text{He normal}}
\sim\mathcal N\left(0,\frac2{\text{fan\_in}}\right)
$$

$$
W_{\text{He uniform}}
\sim\mathcal U\left(
-\sqrt{\frac6{\text{fan\_in}}},
\sqrt{\frac6{\text{fan\_in}}}
\right)
$$

Leaky ReLU 要把负半轴斜率纳入 gain；卷积层计算 fan 时也要包含卷积核覆盖的连接数。这些公式的目的，是让前向信号和反向梯度的尺度更容易保持稳定，而不是保证训练一定成功。

### C.5 随机初始化中的两种分布

如果希望把概率符号也对应到公式，正态分布的密度为：

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

均匀分布在区间 $$(a,b)$$ 内的密度为 $$1/(b-a)$$，区间外为 0；连续分布的端点取值不影响概率。这里的密度不是“恰好抽到某个小数的概率”，区间上的密度积分才是对应概率。


<a id="image-credits"></a>

## 数据与图片说明

本文的函数曲线、结构图和训练图共 18 张，区分了示意数据和实际实验。图 16 使用 MNIST 与 Fashion-MNIST 官方测试集各自前 6 张图片，预测来自配套脚本训练并重新加载的模型，没有按预测是否正确挑选样本。

- MNIST 由 Yann LeCun、Corinna Cortes、Christopher J. C. Burges 整理，通过 [torchvision 的 MNIST 接口](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.MNIST.html)读取。
- Fashion-MNIST 由 Han Xiao、Kashif Rasul、Roland Vollgraf（Zalando Research）发布，见[数据集仓库](https://github.com/zalandoresearch/fashion-mnist)。[MIT 许可]({{ '/assets/img/posts/deep-learning/fashion-mnist-LICENSE.txt' | relative_url }})随下载包保留。

下载包不包含完整数据集、Python 环境或训练权重。复制图片时，请一并保留 `assets/README.md` 和许可文件。
