# Jev 判断模型学习示例

正文源码：[Jev 入门指南](../../_posts/2026-10-23-jev-for-backend-engineers.md)。

本文配套三个接口的请求文件、无需第三方依赖的 Python 示例，以及三张原理图。复审结果与 38 项知识点核对见 [review.md](review.md)。

## 离线运行

进入本目录；如果使用下载包，先解压并进入 `jev-demo/`：

```bash
python3 examples/decision_demo.py
```

预期输出：

```text
候选概率： [0.844, 0.114, 0.042]
工单去向： billing_queue
```

默认示例使用构造数据，不加载模型、不调用网络。

## 在线请求与响应读取

自行取得 TypeSafe API Key，并通过环境变量设置 `TYPESAFE_API_KEY`。进入示例子目录后运行：

```bash
cd examples
bash call_jev.sh
python3 decision_demo.py --response response.json
```

需要 curl 7.76 或更新版本。请求来自 `request.json`，响应保存在脚本同目录的 `response.json`。Python 程序读取其中的 `answers.team`，再应用教学分流策略。

默认阈值仅作演示；生产使用需验证阈值、检查响应、处理超时并配置降级。不要提交 API Key 或真实响应内容。

## 文件与验证范围

- `examples/decision_demo.py`：softmax 与工单分流，支持读取保存的响应。
- `examples/request.json`：包含 Choice、Score、Noul 的请求。
- `examples/call_jev.sh`：在线调用脚本。
- `review.md`：独立审校记录，不属于博客正文。
- 三张图片维护在 `../../assets/img/posts/jev/`，打包时放入 `assets/`。

Python 默认模式、路由分支和响应读取做过离线验证；响应读取使用构造的 API 格式数据。调用脚本通过 shell 语法检查，未使用真实凭据调用 Jev，也未验证开源模型效果。
