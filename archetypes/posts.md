---
title: "{{ replace .File.ContentBaseName "-" " " | title }}"
date: {{ .Date }}
draft: true
tags: []
# categories 只能从固定清单选（见 agent_skills/write-post.md）：算子开发 / 训练框架 / 推理框架 / LLM算法 / 多模态算子 / 强化学习 / Linux / C++ / Python / 随笔
categories: []
# 系列文章必须填 weight（阅读顺序从 1 起），否则同日期下中文序号会按 Unicode 码点错乱排序
# weight: 1
---
