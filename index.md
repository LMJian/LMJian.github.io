---
layout: home
title: 首页
---

# 欢迎来到我的博客

这里是我记录技术学习、工作思考和生活点滴的地方。

## 最新文章

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <small>{{ post.date | date: "%Y-%m-%d" }}</small>
    </li>
  {% endfor %}
</ul>
