# 李明健的博客

基于 GitHub Pages + Jekyll + Chirpy 主题搭建的个人技术博客。

## 访问地址

https://LMJian.github.io/

## 技术栈

- **GitHub Pages** - 静态网站托管
- **Jekyll** - 静态站点生成器
- **Chirpy** - 博客主题
- **GitHub Actions** - 自动构建部署

## 本地预览

```bash
# 安装依赖
bundle install

# 启动本地服务器
bundle exec jekyll serve

# 访问 http://localhost:4000
```

## 写新文章

在 `_posts/` 目录下创建 `YYYY-MM-DD-标题.md` 文件：

```markdown
---
title: "文章标题"
date: 2026-09-15 10:00:00 +0800
categories: [分类]
tags: [标签1, 标签2]
---

文章内容...
```

## 部署

推送到 `main` 分支后，GitHub Actions 会自动构建并部署。
