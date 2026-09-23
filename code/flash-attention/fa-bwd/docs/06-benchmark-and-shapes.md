# 对标分析：FA2 / FA3 / TE 到底谁快，以及目标形状（GQA/MQA/MLA）

> 这一篇回答三个问题：
> ① 反向到底该和谁比？（结论：H100 上要拿 **FA3（SM90）**，不是 pip 里的 FA2）
> ② 三个实现（FA2 / FA3 / TE）在同一批形状上分别多少**时间**、谁赢？
> ③ 我们（ours）离它们还有多远。
>
> 基准一律是**纯反向**（forward 不计入计时）、CUPTI 纯 device 时间，单位 **ms/µs**（越小越好）。
> `TF/TFLOPS` = 每秒 Tera 次浮点运算，是**吞吐**、不是时间，仅作参考。硬件 H100（sm90）。

---

## 1. 先讲清「和谁比」：FA2 是 Ampere 内核，FA3 才是 H100 内核

`flash_attn` 有两个代际：

| 包/来源 | 代际 | 用什么算 | 在 H100 上 |
|---|---|---|---|
| `pip install flash-attn` → **2.7.4** | **FA2 / SM80** | `mma.sync` + `ldmatrix` + `cp.async` | 能跑，但没用 Hopper 的 wgmma/TMA |
| 仓库 `hopper/` → **`flash_attn_3`** | **FA3 / SM90** | **`wgmma` + TMA + warpgroup** | H100 的正解 |

所以「FA 比 TE 慢」这个现象只在拿 **FA2** 比时成立；**换成 FA3，FA 反而比 TE 快**（见 §3）。
**对标一定要选同代实现，否则结论会反过来。**

> FA3 不是 pip 里能装到的，需要从源码编译；本文的 FA3 = 从 `~/github/flash-attention/hopper`
> 用本地 CUTLASS 编译出的 `flash_attn_3 3.0.0`（只编 HDIM128/SM90，11 个目标）。编译细节见 §5.4。

---

## 2. 基准口径：只测「纯反向」

一个容易混的点：如果计时区间里既有 forward 又有 backward，测到的是**两者之和**。
本文统一把 forward 移出计时区：

- FA2/FA3：`autograd.grad(o, [q,k,v], do, retain_graph=True)`；
- TE：先算好 `aux_ctx`，只反复跑 `fused_attn_bwd`。

工具：`harness/fa_vs_te_bwd_only.py`（CUPTI 纯 device 时间，同 shape / 同 dtype / 同 causal）。
因此下表的数字可以直接横向比。

---

## 3. 三方对比：FA3 > TE > FA2（纯反向）

单位 **µs**（越小越快）；fp16 与 bf16 结果一致。

| 形状 (B,S,H,D) | FA2.7.4 (SM80) | **FA3 (SM90)** | TE2.14 (cuDNN SM90) | FA3/FA2 |
|---|---|---|---|---|
| (1,1024,40,128) kv=8 GQA | 186.9 µs | **121.1 µs** | 132.0 µs | 1.54× |
| (1,1024,32,128) kv=4 GQA | 158.6 µs | **82.2 µs** | 112.1 µs | 1.93× |
| (1,1024,64,128) kv=4 GQA | 266.5 µs | **159.4 µs** | 193.5 µs | 1.67× |
| (1,1024,64,128) kv=1 MQA | 266.5 µs | **156.8 µs** | 214.2 µs | 1.70× |
| (1,4096,16,128) MHA | 728.2 µs | **323.3 µs** | 444.6 µs | 2.25× |
| (1,256,2,512) MLA | 不支持 | 不支持 | 不支持 | — |
| (1,512,4,512) MLA | 不支持 | 不支持 | 不支持 | — |
| (1,1024,2,512) MLA | 不支持 | 不支持 | 不支持 | — |

**怎么读这张表**：

- **FA3 最快**：GQA/MQA 比 TE 快 ~8–36%，大 MHA 快 ~38%（323 vs 445 µs）。
- **FA2 最慢**：比 FA3 慢 1.5–2.25×，因为它是 Ampere 内核。
- **MLA（head_dim=512）三者都不支持反向**：FA2/FA3 限制 head_dim ≤ 256，TE 训练反向也限 ≤ 256。
  这三个形状的反向数字**只能由我们自己的实现给出**（见 `docs/04` §7）。

---

## 4. 目标形状重点分析（GQA / MQA / MLA）

### 4.1 数值（vs fp32 参考实现，max_abs）

- fp16 容差约 1e-2、bf16 约 1e-1；FA2/FA3/TE 与 ref 同量级（逐形状表见 `docs/04` §7.1）。
- **规律**：KV 头越少（MQA），`dk/dv` 误差越大——因为每个 KV 头要承载 `H/Hkv` 个 Query 头的梯度，
  低精度的误差在更多项上叠加。

### 4.2 性能上的形状效应

- **GQA/MQA（D=128）**：FA3 82–159 µs，TE 112–214 µs。KV 头越少，FA3 相对 TE 的优势越大。
- **MLA（head_dim=512）**：三个实现都不支持反向；ours 的 fp16/bf16/fp8 都已支持并给出数字
  （`docs/04` §7.7）。

---

## 5. 为什么有快慢差别：kernel、指令、SOL 三层证据

### 5.1 跑的根本不是同一代 kernel（kernel 名单）

以 (1,4096,16,128) 一次反向为例：

| | 反向主 kernel | 时间 |
|---|---|---|
| FA2 | `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<...>>` | 683 µs |
| FA3 | `flash::enable_sm90<flash::FlashAttnBwdSm90<flash::CollectiveMainloopBwdS...>>` | **283 µs** |
| TE | `cudnn_generated_..._sm90_flash_bprop_`**`wgmma`**`_f16_...` | 397 µs |

FA2 的名字里是 `Flash_bwd_kernel_traits`（SM80 模板）；FA3/TE 是 SM90 的 `wgmma` 内核。

### 5.2 指令集（SASS）证据

用 `ncu --page source --print-source sass` 统计主反向 kernel 的指令：

| 指令 | FA2 | **FA3** | TE | 含义 |
|---|---|---|---|---|
| `HMMA.16816.F32` | 320 | 0 | 0 | Ampere `mma.sync` |
| `LDSM`（ldmatrix） | 144 | 0 | 0 | 共享内存取片段 |
| `LDGSTS`（cp.async） | 32 | 0 | 0 | Ampere 异步拷贝 |
| `UTMALDG/UTMASTG/UBLKCP` | 0 | **20/4/8** | 8/4/– | **TMA** |
| `WARPGROUP.ARRIVE/DEPBAR` | 0 | **5/5** | 5/5 | **wgmma** 同步 |
| `STSM`（stmatrix） | 0 | 20 | 20 | 片段回写 |

**FA3 与 TE 是同一代（TMA + wgmma + stmatrix），FA2 是上一代（mma.sync + cp.async）。**
这就是时间差的根因。

### 5.3 ncu SpeedOfLight / occupancy（主反向 kernel，S=4096）

| 指标 | FA2 | **FA3** | TE |
|---|---|---|---|
| Duration | 682 µs | **289 µs** | 399 µs |
| Compute (SM) | 39.8% | **67.6%** | 46.7% |
| DRAM Throughput | 6.6% | 15.4% | 11.3% |
| L1/TEX Throughput | 72.2% | 75.6% | 63.2% |
| Achieved Occupancy | 12.5% | 15.3% | 15.6% |

FA3 的 `Compute` 利用率最高（67.6%）——张量核喂得最满，所以最快。三者都**不是 HBM/算力 bound**，
是片上吞吐 + 延迟受限。

### 5.4 FA3 怎么编译（复现用）

- 源码 `~/github/flash-attention/hopper/`；CUTLASS 用本地 `~/github/cutlass` 符号链接到 `csrc/cutlass`
  （submodule 拉不动），并 `git config submodule.csrc/cutlass.update none`。
- 只编需要的实例（否则 451 个 `.cu` 太久）：
  `FLASH_ATTENTION_FORCE_BUILD=TRUE` + `DISABLE_SM80` + `DISABLE_HDIM64/96/192/256` +
  `DISABLE_HDIMDIFF64/192` + `DISABLE_LOCAL/SOFTCAP/PAGEDKV/APPENDKV/FP8`，`MAX_JOBS=64`。
  实际只编 **11 个目标**，几十秒完成；`import flash_attn_3` 得到 3.0.0。
- 数值验证（`harness/fa3_probe.py`）：与 fp32 ref 的 o/dq/dk/dv 差在 fp16 噪声量级。

---

## 6. 我们（ours）离它们有多远

ours 从「标量实现」起步，已一路优化（详见 `docs/08` 调优历程）。当前（纯反向口径，端到端）：

| shape | ours | FA3 | 时间比 | 备注 |
|---|---|---|---|---|
| (1,4096,16,128) fp16 | ~1.95 ms | 0.324 ms | ~6.0× | 已有 `mma`+`cp.async`，墙 = L2+occupancy |
| (1,1024,64,128) kv4 fp16 | ~0.377 ms | 0.082 ms | ~4.6× | 已支持 GQA |
| (1,4096,16,128) fp8 | ~3.40 ms | —（FA3 无 FP8 bwd） | 对 TE ~7.5× | mma 张量核 |
| MLA (head_dim=512) | 仅有 ours | 三者都不支持 | — | ours 是唯一能跑的 |

差距主要在 **wgmma + TMA 的完整数据通路与 occupancy**（下一步 O9）。这是「同代对标」告诉我们的方向。

---

## 7. 方法论小结（这篇最想留下的）

1. **对标要选同代**：FA2（Ampere）不能代表 FA 在 H100 的实力，要拿 **FA3**。
2. **口径要一致**：纯反向 vs 正向+反向，差一个 forward（FA2 的 forward 尤其慢），结论会变。
3. **单位统一用时间**：TFLOPS 是吞吐，容易看反；本文以 µs/ms 为准。
4. **能力边界要写清**：MLA head_dim=512 三套参考都不支持反向，ours 是唯一能提供数字的。
