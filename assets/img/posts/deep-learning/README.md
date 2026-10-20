# 图片说明

文件名以 `01`～`11`、`13`～`18` 开头的图片是为本文重新绘制的教学图，包含示意图、数学函数曲线与线性回归实际实验曲线。

`12-predictions.png` 的上排展示 MNIST 官方测试集前 6 张图，下排展示 Fashion-MNIST 官方测试集前 6 张图。标签和预测由配套实验生成，没有按照预测是否正确挑选样本。

- MNIST：手写数字数据集，由 Yann LeCun、Corinna Cortes、Christopher J. C. Burges 整理。本文使用 [torchvision 的 MNIST 数据集接口](https://docs.pytorch.org/vision/0.24/generated/torchvision.datasets.MNIST.html)读取，并保留内置文件校验。
- Fashion-MNIST：Han Xiao、Kashif Rasul、Roland Vollgraf，Zalando Research。[数据集与说明](https://github.com/zalandoresearch/fashion-mnist)。其原始 [MIT 许可](fashion-mnist-LICENSE.txt)随图片保留。

本文没有重新分发完整数据集。复制或发布博客时，请将此说明和许可文件与 `12-predictions.png` 一起保留。

正文图号按阅读顺序排列，因此不一定与文件名前缀相同。
