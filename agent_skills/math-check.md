# math-check — 公式渲染自检

## 触发场景

- 文章包含 LaTeX 公式（`$...$` 或 `$$...$$`），发布前自检；
- 用户反馈页面上公式显示异常（出现原始 `\sum` 文本、缺下标、成对消失等）。

## 前置条件

- `hugo.toml` 中以下配置必须存在（2026-08-28 修复，删除即复发）：

```toml
[markup.goldmark.extensions.passthrough]
  enable = true
  [markup.goldmark.extensions.passthrough.delimiters]
    block = [["$$", "$$"], ["\\[", "\\]"]]
    inline = [["$", "$"], ["\\(", "\\)"]]
```

## 症状对照表

| 页面症状 | 根因 | 修法 |
|---|---|---|
| 公式源码原样显示（`\sum_{k...` 可见）且 `_` 消失 | passthrough 配置缺失，Goldmark 把 `_` 当斜体吃掉 | 恢复上面的 toml 配置 |
| `aligned` 块显示红色错误 | KaTeX 不支持 `\boxed{\begin{aligned}...}` 嵌套 | 去掉 `\boxed`，用裸 `aligned` |
| 行内公式渲染成块级或反之 | 分隔符配置错（`$` 必须是 inline） | 核对 delimiters 配置 |
| 公式完全没渲染（整段文本） | KaTeX 资源 404（static/katex 缺失或路径错） | 检查 `static/katex/katex.min.css` 存在且线上 200 |

## 自动自检脚本

构建后运行（在仓库根目录）：

```bash
/tmp/hugo_bin/hugo --gc --minify
python3 - <<'EOF'
import re, glob
bad = 0
for f in sorted(glob.glob('public/posts/*/index.html')):
    html = open(f).read()
    body = html[html.find('<article'):html.find('</article')]
    # 1. markdown 把 _ 吃成 <em> 的证据：数学分隔符内出现 <em>
    n_em = len(re.findall(r'\$[^$]*<em>[^$]*\$', body))
    # 2. 下标被吃掉的证据：\sum{ 形式（应为 \sum_{）
    n_strip = len(re.findall(r'\\sum\{|\\max\{|\\min\{', body))
    if n_em or n_strip:
        bad += 1
        print(f'{f}: em-in-math={n_em} stripped-subscript={n_strip}')
print('PASS' if bad == 0 else f'FAIL: {bad} file(s)')
EOF
```

## 验收标准

- 自检脚本输出 `PASS`；
- 线上页面用浏览器抽查至少一个含 `aligned` 的块级公式和一个含 `_` 下标的行内公式。

## 已知坑

- 检查 HTML 时先定位 `<article>` 正文——页面 `<script type=application/ld+json>` 里的 SEO 摘要也会有公式文本且带 JSON 转义（`\\`、`\\n`），那是正常的，不要误判。
- 本机无 node，无法离线跑 KaTeX 编译验证；以 HTML 字节级检查 + 线上抽查为准。
