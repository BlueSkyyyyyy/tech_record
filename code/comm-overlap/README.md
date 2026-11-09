# comm-overlap：通算并行的 CPU 模型

配套[《通算并行研读》](../content/posts/comm-overlap-00-index/index.md)系列。

**没有 GPU 也能把账算清楚。** 这里的 7 个文件全是纯 Python 标准库、无第三方依赖，
每个都能 `--test` 自测；输出是**解析模型/离散事件模拟**，不是实测。
论文里的实测数字在文章正文中单独标注。

## 文件

| 文件 | 作用 | 对应文章 |
|---|---|---|
| `overlap_model.py` | 统一性能账本：PP / DP / TP / MoE 的解析式；`Ledger` 同时给 exposed、overlap efficiency、bubble rate | 一 |
| `schedule_sim.py` | 流水线调度模拟器：GPipe / 1F1B / 1F1B-split / ZB-H1 / ZB-H2；`--iters` 跨步稳态、`--barrier` 单步口径、ASCII Gantt | 一、二 |
| `bucket_sim.py` | DP 梯度 bucket：尾延迟 vs launch 开销；单通道闭式 + 多通道 | 四 |
| `moe_overlap.py` | MoE 三段流水：dispatch / expert GEMM / combine；扫描通信 SM 数与 chunk 数 | 六 |
| `sm_split_model.py` | 通信/计算 SM 分区：带宽饱和 + wave 量化 | 七 |
| `decode_overlap.py` | 推理 token 切分的收益阈值：小 batch 为什么越切越慢 | 九 |
| `overlap_audit.py` | 从 trace CSV 算 overlap ratio / exposed / per-stream busy | 十 |

> 2026-11 补遗（总览 + UniEP/MegaFlux/Entwine·mKernel 三篇深读）是论文拆解，不新增模型；
> 现有 7 个模型仍适用，补遗里引用的实测数字都标了硬件/配置。

## 快速开始

```bash
python3 overlap_model.py --test
python3 schedule_sim.py --test
python3 bucket_sim.py --test
python3 moe_overlap.py --test
python3 sm_split_model.py --test
python3 decode_overlap.py --test
python3 overlap_audit.py --test
```

一次跑全部：

```bash
for f in *.py; do echo "== $f =="; python3 "$f" --test || exit 1; done
```

## 几个可以直接看的输出

```bash
# 四种排法在同一组参数下的气泡（p=4, m=8, F=B=W=1）
python3 schedule_sim.py --stages 4 --microbatches 8 --schedule 1f1b        # 33, 气泡 27.3%
python3 schedule_sim.py --stages 4 --microbatches 8 --schedule 1f1b-split  # 30, 气泡 20.0%
python3 schedule_sim.py --stages 4 --microbatches 8 --schedule zbh1        # 27, 气泡 11.1%
python3 schedule_sim.py --stages 4 --microbatches 8 --schedule zbh2 --iters 6  # 24.5/步, 2.0%

# ZB-H2 的零气泡只在跨步成立：加 --barrier 就退回单步下界
python3 schedule_sim.py --schedule zbh2 --iters 6            # per_step=73.17
python3 schedule_sim.py --schedule zbh2 --iters 6 --barrier  # per_step=79.00

# 看 Gantt
python3 schedule_sim.py --schedule zbh1 --stages 4 --microbatches 8 --gantt

# DP bucket 的凹陷
python3 bucket_sim.py --sweep --launch 2

# MoE：通信 SM × chunk 数
python3 moe_overlap.py --sweep

# SM 分区最优
python3 sm_split_model.py

# decode 阈值
python3 decode_overlap.py --scan

# trace 审计
python3 overlap_audit.py --demo
```

## 模型假设（读输出前先看）

- **单位**：`overlap_model.py` / `schedule_sim.py` / `bucket_sim.py` 用无量纲时间；
  `moe_overlap.py` / `sm_split_model.py` / `decode_overlap.py` 用微秒（`us`）。
- **流水线**：F/B/W 是固定时长；同一 stage 串行；跨 stage 依赖用拓扑式 list scheduling。
  五种排法的区别只有两件事：**warm-up 做几个 F**（决定在飞数）和 **wgrad 允许积压几个**（决定 W 排在哪）。
  `zbh1` / `zbh2` 是按论文 Table 2 的形状手写的静态表，不是论文的自动调度（ZB-1p / ZB-2p）；
  等时假设下气泡分别落在 $(p-1)(T_F+T_B-T_W)$ 与 $(p-1)(T_F+T_B-2T_W)$，
  但后者只有在**跨 step 稳态**（`--iters > 1` 且不加 `--barrier`）才能达到。
- **DP bucket**：桶按反向顺序就绪；通信通道 FIFO；`launch` 是每桶固定开销。
- **MoE**：一个通信通道 + 一个计算通道；dispatch/combine 共享通信通道；
  计算时间按 SM 数线性 scaling，不含真实 GEMM 的 tile 效率。
- **SM 分区**：`T_compute = Wc/(S-s)`，`T_comm = Wm/min(s,s0)`；
  `wave_unit` 是粗粒度的 wave 量化惩罚。
- **decode**：`c(T)=max(ceil(T*w/S/wave)*wave, wave)`，`m(T)=T*v/BW`；
  split 用两级流水 `max(2c_half, 2m_half)+min(c_half,m_half)`。
- **trace 审计**：区间并集/交集，不做 stream 间依赖推断；输入需要自己从 nsys 导出。

## 为什么用 CPU 模型

1. **可复现**：不依赖 GPU 型号、驱动、NCCL 版本；换参数就能扫。
2. **可解释**：每个公式都能手算，能看出"哪个假设导致了这个结论"。
3. **可迁移**：真实调优时先用模型缩小区间，再到 GPU 上验证，省时间。

但模型不能替代实测：

- 真实 GEMM 的 tile 效率、wave 量化、L2 命中率都比线性 scaling 复杂；
- NCCL 的 ring/tree 算法、消息大小、通道数会影响有效带宽；
- overlap tax 与功耗/频率相关，模型里没有。

**用模型找方向，用 `ncu`/`nsys` 定结论。**

## 和文章的关系

| 文章 | 用到的文件 | 关键结论 |
|---|---|---|
| 一：性能账本 | `overlap_model.py`、`schedule_sim.py`、`bucket_sim.py` | $T=C+E+B$；overlap tax；$\max(C,M)$ 上界 |
| 二：流水线 | `schedule_sim.py` | $p=8,m=24$：1F1B 22.58% → 拆开 W 紧跟 16.28% → ZB-H1 8.86% → ZB-H2 跨步 1.59% |
| 四：DP | `bucket_sim.py` | 桶太大尾延迟、太小 launch 反噬；最优在 $t\approx c$ |
| 六：MoE | `moe_overlap.py` | 通信 SM 太少/太多都慢；chunk 不是越多越好（真实系统有 overhead） |
| 七：Kernel/SM | `sm_split_model.py` | 带宽饱和后多给 SM 只偷计算；wave 量化造成阶梯 |
| 九：推理 | `decode_overlap.py` | 小 batch 切分更慢；模型阈值 ~4K，TokenWeave 用融合降到 1K |
| 十：测量 | `overlap_audit.py` | 用并集/交集算 overlap，避免 kernel 时长重复计数 |

## 许可

与仓库其他代码一致，仅用于学习/研究。
