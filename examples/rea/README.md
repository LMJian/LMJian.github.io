# REA 学习包

[阅读文章](../../_posts/2026-10-23-rea-for-backend-engineers.md)。配图位于站点的 `assets/img/posts/rea/`，下载包中位于 `assets/`。

## 运行教学样本

建议使用符合 REA 6.4.0 声明范围的 Node.js：22.19+ 的 22.x、24.11+ 的 24.x，或 26+。
在本目录执行：

```bash
node examples/check-demo.mjs
```

这会验证运费阈值两侧，并确认负数、小数和不安全整数会被拒绝。不需要 npm 安装、模型账号或分析后端，不调用 REA。

`examples/owned-app/` 是自建 JavaScript 样本。文章还提供固定版本的 REA 静态分析命令及 Codex / DeepSeek Harness 接入说明；完整 REA CLI/MCP 工具链尚未实测通过，不能把业务样本通过视为接入成功。

## 配置片段

- `examples/codex-rea.toml`：Codex MCP 配置片段。
- `examples/deepseek-rea.patch.yml`：DeepSeek Harness patch 片段。

按文章检查作用范围后合并，保留已有配置；这些文件不会自动安装或更改真实客户端。

## 维护下载包

在仓库根目录执行：

```bash
python3 scripts/package_agent_learning_demos.py
python3 scripts/package_agent_learning_demos.py --check
```

下载包从站点文章生成离线 Markdown，并包含三张图、代码、配置片段和 [复核说明](review.md)。修改文章或资源后需要重新生成。
