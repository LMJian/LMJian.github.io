# 提示词工程学习示例

正文源码：[后端工程师的 9 个提示词技巧](../../_posts/2026-10-23-prompt-engineering-for-backend-engineers.md)。知识点覆盖与复审记录见 [review.md](review.md)。

## 运行最小示例

需要 Python 3.10 及以上版本。进入本目录；使用下载包时，先解压并进入 `prompt-engineering-demo/`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/validate_triage.py
```

示例使用构造的 JSON 响应，只做本地校验，不调用在线模型，不需要 API Key。

预期输出是四个字段的 JSON：`category` 为 `upstream_timeout`，证据编号为 `L1`，并包含缺少的信息与下一步检查。

只把 `raw` 中 `evidence_ids` 的 `L1` 改成 `L999`，保持 `logs` 不变，再次运行会触发“响应引用了输入中不存在的日志编号”。

## 文件说明

- `examples/validate_triage.py`：数据结构与证据编号校验。
- `examples/triage.schema.json`：从同一个 Pydantic 模型导出的 JSON Schema；具体模型接口可能仅支持其中部分关键字。
- `requirements.txt`：本次验证使用的 Pydantic 版本。
- `review.md`：9 个知识点的覆盖检查与整篇复审记录。
- 3 张示意图维护在 `../../assets/img/posts/prompt-engineering/`，打包时收入 `assets/`。

本地检查不能证明分类在语义上正确：存在的日志编号也可能被引用到错误结论上。模型效果需要另外使用业务评估集验证。
