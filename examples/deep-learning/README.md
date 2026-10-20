# 深度学习：配套代码与图解

本目录对应博客《写给后端工程师的深度学习入门：从张量到能训练的神经网络》。
包含 5 个完整脚本，以及固定版本的依赖和一次 CPU 实验的参考指标。

## 运行环境

使用 Python 3.12。下列命令在此目录中运行；如果下载了 ZIP，
先解压并进入 `deep-learning-demo/`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r examples/requirements.txt
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。
安装命令使用平台默认 PyPI 包；如果需要 Linux/Windows 的 CPU 专用包、
CUDA 或 ROCm 包，请根据文章第 2 节选择对应的官方安装命令。
固定版本用于复现，不表示最新版。

## 依次运行

```bash
python examples/01_tensor_basics.py
python examples/02_autograd.py
python examples/03_linear_regression.py
python examples/04_vision.py --task mnist --epochs 5 --limit 12000
python examples/05_house_prices.py
python examples/04_vision.py --task fashion --epochs 5 --limit 12000
```

张量、自动微分、线性回归和合成房价示例不需要下载数据。
图像脚本首次运行需要联网获取公开数据集。

默认使用 CPU。图像脚本支持 `--device cuda`、`--device mps`，
但参考结果只验证了 CPU。`--limit 0` 使用完整训练池；
`--epochs`、`--data-dir`、`--output-dir` 分别控制轮数、数据目录和结果目录。

## 结果和边界

输出写入 `runs/linear/`、`runs/mnist/`、`runs/fashion/`、`runs/house/`。
线性回归和图像示例会保存并重新加载权重，核对预测是否一致。
房价示例使用合成数据，只演示训练与评估，未封装可部署的预处理器与模型。

`examples/expected-results.json` 记录一次固定种子的 CPU 实验结果，
可以帮助检查输出数量级；它不是跨硬件、跨版本逐位一致的保证。
MNIST 与 Fashion-MNIST 分别得到 93.08% 和 83.87% 的测试准确率。
小规模教学结果不能代替目标业务数据上的评估。

## 图像数据下载

MNIST 使用 Google 托管的 HTTPS 镜像，Fashion-MNIST 使用数据发布方仓库。
torchvision 会核对内置校验值。

若 Python 下载受网络限制，可以将对应的四个 gzip 文件放到
`data/MNIST/raw/` 或 `data/FashionMNIST/raw/`，再次运行脚本。
文件名见脚本所用数据集的 `resources` 字段。

原实验因本机 Python 下载连接停滞，通过 CLI 下载了同一组 HTTPS 文件，
核对校验值后再运行训练。首次自动下载、GPU 训练不属于已验证范围。

## 图片与打包

下载包的 `assets/` 含 18 张图片、图片说明和 Fashion-MNIST 许可。
在 Git 仓库中，图片统一维护在 `assets/img/posts/deep-learning/`；
打包脚本将它们放入 ZIP，避免在仓库中保存两份相同图片。

在仓库根目录运行：

```bash
python3 scripts/package_deep_learning_demo.py
python3 scripts/package_deep_learning_demo.py --check
```

打包使用固定文件清单，不包含数据集、模型、虚拟环境或训练结果。
