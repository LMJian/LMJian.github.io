# Codex 如何管理“短期记忆”，它和 Claude Code 的差别

- [文章源码](../../_posts/2026-10-22-codex-context-memory.md)
- [教学配图](../../assets/img/posts/codex-context-memory/)
- [覆盖与技术核验](coverage-check.md)
- [离线示例](examples/context_demo.py)

在此目录执行：

```bash
python3 examples/context_demo.py
```

需要 Python 3，无第三方依赖，也不调用模型接口。示例使用人工摘要和自定义事件格式，仅演示活动上下文的重建，不用于评估真实产品的 token 压缩率或摘要质量。

仓库中的 `examples/` 不发布到站点；文章通过 `assets/downloads/codex-context-memory.zip` 提供完整学习包。在仓库根目录重新打包并校验：

```bash
python3 scripts/package_codex_context_demo.py
python3 scripts/package_codex_context_demo.py --check
```
