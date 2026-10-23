# Personal Agent 学习包

[阅读文章](../../_posts/2026-10-23-personal-agent-for-backend-engineers.md)。配图位于站点的 `assets/img/posts/personal-agent/`，下载包中位于 `assets/`。

## 运行最小示例

需要 Python 3.10+，不需要第三方依赖、模型 API 或账号凭证。在本目录执行：

```bash
python3 examples/task_state.py task-state.json
python3 examples/task_state.py task-state.json
```

第一次生成模拟日报并保存 `awaiting_approval`；第二次读取同一状态，不重复生成草稿。
示例只写本地 JSON，不发送邮件。任务日期固定，仅演示单进程状态保存与重复执行识别；不提供并发控制或外部副作用的完整幂等保证。

## 维护下载包

在仓库根目录执行：

```bash
python3 scripts/package_agent_learning_demos.py
python3 scripts/package_agent_learning_demos.py --check
```

下载包从站点文章生成离线 Markdown，并包含三张图、示例和 [复核说明](review.md)。修改文章、配图或示例后需要重新生成。
