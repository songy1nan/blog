# 还真是

Astro + Markdown personal blog for GitHub Pages.

## 本地预览

```bash
npm install
npm run dev
```

## 写文章

在 `src/content/blog/` 里新建 Markdown 文件，例如：

```text
src/content/blog/my-note.md
```

每篇文章开头需要包含：

```markdown
---
title: "文章标题"
description: "文章摘要"
pubDate: 2026-05-20
tags: ["生活", "观察"]
draft: false
---
```

## GitHub Pages 发布

1. 在 GitHub 创建 `blog` 仓库。
2. 把本项目推送到 `songy1nan/blog`。
3. 打开仓库的 `Settings > Pages`。
4. Source 选择 `GitHub Actions`。
5. 推送到 `main` 后会自动发布。

发布地址：

```text
https://songy1nan.github.io/blog/
```
