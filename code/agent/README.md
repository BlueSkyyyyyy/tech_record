# AI Agent 系统工程：配套代码

对应专题 [`content/posts/agent-00-index/`](../../content/posts/agent-00-index/index.md)。
全部纯 Python 标准库、CPU 可跑，每个脚本都有 `--test`。

```bash
cd code/agent
for f in *.py; do echo "== $f"; python3 "$f" --test || exit 1; done
```

| 脚本 | 对应文章 | 演示什么 |
|---|---|---|
| `loop_sim.py` | 第 2 篇 | 剧本模型 + 并行工具 + 错误观察 + `max_steps`；断言并行/串行消息数一致、超步数必停 |
| `mcp_rpc.py` | 第 3 篇 | JSON-RPC 2.0 的 MCP 生命周期：`initialize`、`notifications/initialized`、`tools/list`、`tools/call`、`ping`；子进程 server |
| `token_budget.py` | 第 4 篇 | 原始/中间/字段三种截断策略下的证据保留率与 JSON 合法性；720 次合成 trace |
| `apply_patch.py` | 第 5 篇 | Codex 风格 `*** Begin Patch` 语法、四段模糊定位、多文件非原子失败 |
| `passk.py` | 第 8 篇 | 无偏 `pass@k`、`pass^k`、Wilson 区间、grader bug 对观测值的影响 |
| `gui_ground.py` | 第 9 篇 | 0–1000 归一化坐标、截图缩放误差、点击命中率与 IoU |
| `multiagent_cost.py` | 第 10 篇 | 单 agent / 多 agent / chat 三种 token 账；按 worker 数、输入长度改参数 |
| `grpo.py` | 第 11 篇 | 组内优势、token 级 vs 序列级比率、GRPO/GSPO clip |
| `taint.py` | 第 12 篇 | 逐动作授权 vs 全局 taint 检查；private 数据进外部 sink 的反例 |

## 几个可以自己改的参数

```bash
python3 loop_sim.py --max-steps 2          # 观察预算终止
python3 token_budget.py --runs 50 --budget 800
python3 multiagent_cost.py --workers 10 --worker-in 50000
python3 gui_ground.py --width 3840 --height 2160
python3 passk.py --p 0.75 --k 10
```

## 依赖

无。Python 3.10+。`mcp_rpc.py` 用 `subprocess` 起一个本地 server，不需要网络。
