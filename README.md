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
bundle install

# 启动本地服务器
JEKYLL_ENV=production bundle exec jekyll serve

# 访问 http://localhost:4000
```

提交前验证：

```bash
python3 scripts/check_tracked_files.py
bundle exec ruby scripts/check_home.rb
bundle exec ruby scripts/check_lastmod.rb
JEKYLL_ENV=production bundle exec jekyll build
python3 scripts/check_site.py _site
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

## 部署

推送到 `main` 分支后，GitHub Actions 会自动构建并部署。
