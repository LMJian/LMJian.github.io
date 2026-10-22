# Mason 的博客

基于 GitHub Pages + Jekyll + Chirpy 主题搭建的个人技术博客。

## 访问地址

https://LMJian.github.io/

## 技术栈

- **GitHub Pages** - 静态网站托管
- **Jekyll** - 静态站点生成器
- **Chirpy** - 博客主题
- **GitHub Actions** - 自动构建部署

## 本地预览

使用 Ruby 3.3 和 Bundler（系统自带的 Ruby 2.6 不支持当前主题）。

```bash
# 安装依赖
bundle config set --local path vendor/bundle
bundle install

# 启动本地服务器
JEKYLL_ENV=production bundle exec jekyll serve

# 访问 http://localhost:4000
```

提交前验证：

```bash
python3 scripts/check_tracked_files.py
python3 scripts/package_order_demo.py --check
python3 scripts/package_deep_learning_demo.py --check
python3 scripts/package_nlp_huggingface_demo.py --check
python3 scripts/package_codex_context_demo.py --check
python3 examples/codex-context-memory/examples/context_demo.py
python3 scripts/check_holidays.py
bundle exec ruby scripts/check_holiday_isolation.rb
bundle exec ruby scripts/check_home.rb
bundle exec ruby scripts/check_lastmod.rb
JEKYLL_ENV=production bundle exec jekyll build
python3 scripts/check_site.py _site
python3 scripts/check_holidays.py --site _site
```

`Gemfile.lock` 固定已验证的依赖；升级依赖后重新运行上述检查。

临时报告、验收截图、构建结果、缓存和本地环境变量不应提交。
添加文件到暂存区后、提交前，运行 `python3 scripts/check_tracked_files.py`。
`.gitignore` 不会自动取消已有文件的跟踪；若检查失败，使用
`git rm --cached -- <文件路径>` 取消跟踪并保留本地文件。
GitHub Actions 也会运行此检查；它可以阻止部署，但不能撤回已经推送的文件。

## 写新文章

在 `_posts/` 目录下创建 `YYYY-MM-DD-标题.md` 文件：

```markdown
---
title: "文章标题"
date: 2026-09-15 10:00:00 +0800
categories: [分类]
tags: [标签1, 标签2]
description: "用于首页摘要和搜索引擎的简短介绍"
math: true # 包含公式时启用
media_subpath: /assets/img/posts/文章目录
---

文章内容...
```

图片存放在对应的 `assets/img/posts/` 目录中，正文可使用相对图片名。
文章标题由主题渲染，正文从引言和二级标题开始，避免重复一级标题。

为图片声明实际宽高，给延迟加载的图片预留空间，避免章节跳转后位置漂移。
Kramdown 的图片属性必须紧跟在图片后面，写在同一行：

```markdown
![图片说明](figure.png){: width="1600" height="900" }
```

若将属性单独写在下一行，宽高会落在外层段落上，对图片无效。
构建检查会拒绝缺少宽高的延迟加载图片。

本项目使用 Kramdown。新文章的行内公式建议写成 `$$x_i$$`，由解析器生成
MathJax 的行内定界符；独立公式将 `$$` 各占一行。
绝对值用 `\lvert x\rvert`，避免裸竖线被识别为表格。
构建后检查公式，尤其是撇号、下标和表格中的公式，避免 Markdown 排版改变数学含义。

## 文章配套代码

机器学习文章的可运行项目位于 `examples/order-cancel/`，运行方式见该目录的 README。
修改示例后，重新生成并校验下载包：

```bash
python3 scripts/package_order_demo.py
python3 scripts/package_order_demo.py --check
```

打包使用固定文件清单，不包含训练生成的模型、评估输出或本地虚拟环境。
`examples/` 不直接复制到站点，文章通过 `assets/downloads/order-cancel-demo.zip` 提供下载。

深度学习文章的示例位于 `examples/deep-learning/`，包含 5 个脚本及固定版本依赖。
18 张图统一维护在 `assets/img/posts/deep-learning/`，同时收入下载包：

```bash
python3 scripts/package_deep_learning_demo.py
python3 scripts/package_deep_learning_demo.py --check
```

文章通过 `assets/downloads/deep-learning-demo.zip` 提供代码与图解。
修改源码、图片或说明后，应重新生成 ZIP。

NLP 与 Hugging Face 文章的示例位于 `examples/nlp-huggingface/`，包含 8 个运行入口、
1 个共享模块、教学数据与固定版本依赖。14 张图统一维护在
`assets/img/posts/nlp-huggingface/`：

```bash
python3 scripts/package_nlp_huggingface_demo.py
python3 scripts/package_nlp_huggingface_demo.py --check
```

文章通过 `assets/downloads/nlp-huggingface-demo.zip` 提供完整学习包。
打包脚本从站点文章生成离线 Markdown，将代码与图片链接改为包内路径，
并收入知识点覆盖说明和复核报告。修改文章、源码、图片或说明后，重新生成 ZIP。
Codex 短期记忆文章的配套文件位于 `examples/codex-context-memory/`，包含无第三方依赖的 Python 示例与核验说明。

```bash
python3 examples/codex-context-memory/examples/context_demo.py
python3 scripts/package_codex_context_demo.py
python3 scripts/package_codex_context_demo.py --check
```

文章通过 `assets/downloads/codex-context-memory.zip` 提供离线正文、3 张配图与示例。

GitHub Actions 会检查四个下载包与源文件是否一致。

## 部署

推送到 `main` 分支后，GitHub Actions 会自动构建并部署。

## 卷轴年度调休数据

`assets/holidays/cn/index.json` 是客户端版本索引，`YYYY.vN.json` 是不可变的年度文件。这里只托管静态 JSON，不新增下载网页，不修改博客首页、导航、搜索或文章。数据依据国务院通知整理，当前包含 2026 年的 33 个休息日和 6 个补班日；发布流程及协议见仅供仓库维护的 [数据约定](_docs/holiday-data.md)。

运行 `python3 scripts/check_holidays.py` 校验独立日期对照、哈希和实际本地 HTTP 下载。Pages CI 同时校验源文件与 `_site` 中的文件，并对比有无年度文件时的站点构建结果，确认 JSON 不影响网页。本地通过不代表线上已发布，发布后须下载线上文件核对。App 首次安装使用内置副本，后续下载失败仍保留已有数据。
