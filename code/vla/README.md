# VLA 精读配套代码

对应专题：[VLA 精读](../../content/posts/vla-00-index/index.md)。

七个纯标准库 / NumPy / PyTorch CPU 脚本。它们不是模型实现，而是把文章里的账**算出来**：
分箱误差、mask 形状、flow 与 L2 的分布差异、归一化的离群点、chunk 的断档条件、RTC 的拼接跳变、置信区间。

| 文件 | 依赖 | 对应文章 | 输出要点 |
|---|---|---|---|
| `action_binning.py` | numpy | 一 | 256 条边界 = 255 个中心；最大量化误差 1/255；DCT 保留一半系数 RMSE 不变；delta 累积误差 |
| `block_attn_mask.py` | numpy | 四 | π0 三块 mask（prefix/state/action）；π0.5 两块；prefix KV cache 省 6.7×（P=768,S=51） |
| `flow_matching_toy.py` | torch | 三 | 双峰动作：flow 54.5/45.5、0% 落中间；L2 77.2% 落中间；单步 flow 也塌 |
| `norm_stats.py` | numpy | 七 | q01/q99 vs mean/std 的覆盖；0.5% 尖峰把正常值压到 \|z\|≤0.5；padding 统计污染 |
| `chunk_latency.py` | numpy | 八 | 同步/异步刷新率；`C*dt >= L` 不断档，`>= 2L` 有余量；对齐索引 vs 不对齐的跳变差 9× |
| `rtc_toy.py` | numpy | 八、九 | 四种 prefix 权重曲线；zeros/ones/linear/exp 的拼接跳变与 chunk 内最大步跳变 |
| `eval_ci.py` | 标准库 | 十一 | Wilson 区间；8/10 vs 6/10 不能区分；区分 60% vs 80% 每组需 83 次；多任务方差主导 |

## 运行

```bash
cd code/vla
for f in *.py; do
  echo "== $f =="
  python3 "$f" --test || exit 1
done
```

`flow_matching_toy.py` 需要 PyTorch（CPU 即可）；其余只需 NumPy 或标准库。

## 几个可复现的数字

```text
# action_binning.py
边界数=256 中心数=255 步长=0.007843
理论最大误差=0.003922  RMS=0.002264
步数= 50  最坏累积=0.1961  RMS 累积=0.0160

# flow_matching_toy.py（歧义区样本）
flow matching   54.5% 左 / 45.5% 右 / 0.0% 中间
L2 回归          0.0% 左 /  0.0% 右 / 77.2% 中间

# chunk_latency.py（C=50, L=200ms, dt=20ms）
同步刷新 0.83 Hz / stall 17%；异步 1.25 Hz / stall 0%
naive 跳变 0.2242；aligned 跳变 0.0242

# eval_ci.py
8/10 = 80% [49%, 94%]；80/100 = 80% [71%, 87%]
60% vs 80%：每组 83 次（80% power）
```

## 说明

- 所有脚本都是**玩具模型**，用来验证量级和趋势，不是论文复现。
- 随机种子固定；`--test` 只断言结构性结论（如「L2 落在中间的比例 > 50%」），不断言具体小数。
- 代码里的 π0/π0.5 行号对应 openpi `215abfb2`、LeRobot `ae32126a`、OpenVLA `c8f03f48`、OFT `e4287e94`。
- 不依赖 GPU、不下载数据、不访问网络。
