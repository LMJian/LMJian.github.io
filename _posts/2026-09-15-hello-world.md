---
title: "Hello World - 我的第一篇博客"
date: 2026-09-15 10:00:00 +0800
categories: [随笔]
tags: [博客, 开篇, GitHub Pages]
pin: true
---

欢迎来到我的博客！这是我用 **GitHub Pages + Jekyll + Chirpy 主题** 搭建的个人博客的第一篇文章。

## 为什么写博客

> 写作是最好的思考方式。

- **记录学习**：把学到的东西整理成文字，加深理解和记忆
- **分享交流**：希望能帮助到有同样问题的人，也欢迎指正
- **倒逼输入**：为了有内容可写，会主动去学习和探索
- **个人品牌**：长期积累，形成自己的技术影响力

## 博客技术栈

这个博客基于以下技术搭建：

| 技术 | 用途 |
|------|------|
| GitHub Pages | 免费静态网站托管 |
| Jekyll | 静态站点生成器 |
| Chirpy | 博客主题（支持暗黑模式、搜索等） |
| Markdown | 文章写作格式 |
| GitHub Actions | 自动构建与部署 |

## 主题功能展示

Chirpy 主题提供了很多实用功能，下面是一些示例：

### 代码高亮

```python
def quick_sort(arr):
    """快速排序算法"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

if __name__ == "__main__":
    data = [3, 6, 8, 10, 1, 2, 1]
    print(f"排序结果: {quick_sort(data)}")
```

### 提示框

> [!NOTE]
> 这是一个提示信息，用于补充说明。

> [!TIP]
> 这是一个建议，帮助你更好地使用博客。

> [!WARNING]
> 这是一个警告，需要特别注意的内容。

### 引用块

> 代码是写给人看的，只是顺便能在机器上运行。
>
> — Harold Abelson

## 后续计划

接下来我会在这里分享：

1. **技术学习笔记** - 后端开发、系统架构、AI 工程化
2. **项目实战经验** - 工作中遇到的问题和解决方案
3. **工具使用技巧** - 提升效率的工具和方法
4. **读书与生活感悟** - 读书笔记和生活思考

## 写在最后

搭建博客只是第一步，坚持写作才是最难的。希望自己能保持更新，也欢迎你常来逛逛。

如果你有任何问题或建议，欢迎通过邮件或 GitHub 与我联系。

感谢你的访问！
