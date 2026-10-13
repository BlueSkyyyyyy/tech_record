# MoE wgrad 的「fp32 物化 / 归约税」：融合进优化器能拉回多少 2.2×

> 本文对应路线图 **P137-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 147 篇。
>
> 第 145 篇（P135-1）量化了 wgrad 慢的根因（每专家一份 fp32 权重梯度），第 146 篇（P136-1）把
> 「wgrad 换 bf16 输出」落地：**真实 SM90 GEMM 本身快 2.27×**，但端到端在 `S≈8–16` 封顶 **~1.55×**，
> 因为「把每步 fresh 的 bf16 梯度再 Σ 成一份 fp32 参数梯度」要付一笔**归约税**。它留下一个**未答问题**：
> **若让优化器直接吃 bf16 梯度、不物化 fp32，能拉回多少？**
>
> 本篇用三个可跑协议把它量清楚，结论有两层：
>
> 1. **融合（优化器直接吃 bf16）确实能收回「物化税」**：端到端从 P136 的 **1.48×**（`S=16`，
>    每步 fresh bf16 + `torch.sum` + 优化器）提到 **1.67×**——正好顶到**理论天花板 1.69×**。
>    关键在于**不再落地那份 470MB 的 fp32 梯度**（省掉「写 + 再读回」共 8B/elem）。
> 2. **但 2.27× 这个 GEMM-only 上界不可达**：消费端（无论优化器还是归约）**至少要把 S 份 per-step
>    bf16 梯度读一遍**（`235MB/步`，实测 ~76us/步）；它叠加在 `219.6us/步` 的 bf16 GEMM 之上，
>    把天花板钉在 `499/(219.6+76)≈1.69×`。**2.27× 隐含「这份读是免费的」——它从来不是。**
>
> 可跑性：全部 SM90 FP8 kernel（H100 可跑）；消费侧内核（reduce / atomic / fused optimizer）
> **不碰 UMMA/tcgen05、纯 bf16 访存 + 浮点，架构无关**，直接编译运行真实设备代码。SM100 的
> wgrad `k_grouped_tn_contiguous` 同族 fp32 累积，**仅分析**。本机 8 卡共享，本文在**空闲 GPU1**
> 单进程测，报数注明。

---

## 1. 合同：三种「把 S 份梯度交给优化器」的协议

先看可选项。第 146 篇已核过，SM90 的 wgrad 入口 `k_grouped_fp8_gemm_nt_contiguous`
（`csrc/apis/gemm.hpp:373-427`）有两条硬约束：`recipe == (1,1,128)`（`:384`）与 **`C` 必填**
（`:403` `c.has_value()`），唯一落点是 **1D1D** kernel，而 1D1D 的 host 契约又把输出钉死为
**fp32 且必累积**（`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`）：

```
DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);
```

落到 kernel 是 `SM90_TMA_REDUCE_ADD_2D`（`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`）——
**每步都要把整份 fp32 `C`（`[E,M,N]`）读回、再加一次、写回**。另一侧 1D2D 只能 **bf16 直写、禁累积**
（`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:81`）：

```
DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16);
```

于是「同一个 wgrad 数学」在 SM90 上只有两种 GEMM 形态（P136 表），而**累积 S 步 + 交给优化器**
这件事有三种协议：

| 协议 | 每步 GEMM | 每步线上字节 | 末次消费 | 消费端字节 |
|---|---|---|---|---|
| **REF** | 1D1D `k_grouped`（fp32/累积） | 读 `C` 4B + 写 `D` 4B = **8B/elem** | 优化器读 fp32 梯度 | 读 4B（+ 读/写 master） |
| **ALT-R** | `E×` 稠密 1D2D bf16 | 写 **2B/elem** | `torch.sum` 成 fp32 | 读 `S·2B` + 写 4B（+ 优化器读 4B） |
| **ALT-OPT** | 同上 | 写 **2B/elem** | **融合**：读齐 S 份 bf16 就地求和、直接更新 master | 读 `S·2B`（+ 读/写 master，**无 fp32 中间**） |

**差异只落在「归约写 fp32 那一下 + 优化器再读回」**（共 8B/elem）——这就是「物化税」。
`ALT-OPT` 把它整段省掉：S 份 bf16 梯度只被读一次，求和发生在优化器的寄存器里。

---

## 2. 方法：真实 GEMM + 手写消费侧下界内核

**基准形状**：`dW2: [E=8, M=2048, N=7168]`（DeepSeek 量级），`sum_k=4096`、每专家 `k_e=512`；
消费端统一用一个 SGD-lite `w ← w − lr·G`（AdamW 只多各自一份状态，对三协议是公共项）。

- **GEMM 段**：真实 DeepGEMM kernel——REF 走 `k_grouped_fp8_gemm_nt_contiguous`（1D1D、`recipe=(1,1,128)`、
  `C=D` 累积），ALT 走 `E×` `fp8_gemm_nt`（1D2D bf16 直写）。
- **消费段**：`src/wgrad_reduce_consumer.cu` 手写四个内核（架构无关，H100 真机）：
  - `k_reduce8`：`bf16[S][G] → fp32[G]`，8 元素/线程、`float4` 向量化读写——**归约下界**；
  - `k_fused_sgd8`：`w ← w − lr·Σ_s g_s`，读 S 份 bf16 + 读/写 w，**不落地 fp32 梯度**——**融合下界**；
  - `k_reduce_atomic`：先清零再用 `atomicAdd` 逐元素累加——证伪「拆成 atomicAdd 更快」；
  - `k_read8`：纯读一份 bf16——给峰值读参照。
  内核正确性用 GPU vs **CPU double** 逐元素对拍（`max|err|=2.384e-07`，即 fp32 求和舍入）。

**理论天花板**：若消费端把每步 bf16 梯度按满带宽读一遍，则渐进端到端上界为
`t_ref / (t_alt + 235MB/BW)`，其中 `t_ref`/`t_alt` 是 REF/ALT 的单步 GEMM、`BW` 是实测消费带宽。

---

## 3. 结果

### 3.1 消费侧：`atomicAdd` 灾难、`torch.sum` 次优、手写 128-bit 核 ~3.08 TB/s

Part A（`harness/exp_wgrad_reduce_consumer.out.txt`）。纯读一份 bf16（235MB）**78us（3009 GB/s）**，
即该机型的**峰值读带宽参照**。消费侧各内核：

| 核 | S=1 | S=4 | S=8 | S=16 |
|---|---:|---:|---:|---:|
| `custom reduce8`（读 `S·2B` + 写 4B） | 288.1us | 472.8 | 762.6 | **1371.8** |
| `custom fused_sgd`（读 `S·2B`+8B） | 396.3 | 611.8 | 914.6 | **1532.1** |
| `torch.sum(dim=0,fp32)` | 381.4 | 569.6 | 837.4 | **1726.8** |
| `custom atomic`（RMW） | **23370** | — | — | — |

三个非平凡读法：

1. **`atomicAdd` 直接被否决**：`S=1` 就 **23.4ms**（`RMW` 把写变成「读-改-写」，且 fp32 随机地址争用），
   比手写 reduce 慢 **~80×**。**不能把「末次归约」拆成逐元素的 atomicAdd**。
2. **`torch.sum` 不是最优**：`S=16` 时 `torch.sum` 1726.8us vs 手写 **1371.8us（1.26×）**、`S=8`
   837.4 vs 762.6us。手写核稳在 **~3.08 TB/s（≈HBM 峰值 3.35 的 92%）**，而 `torch.sum` 大 S 时掉到
   2.45 TB/s。**P136 量到的 838.6us「归约税」用的是一个次优实现**——归约本身还能再挤 ~10–26%。
3. **手写核也有个坑：fp32 store 必须向量化**。首版 reduce 用 8 个标量 `o[j]=a[j]`，`S=1` 只有
   **778us / 905 GB/s**（nvcc 未把标量 4B store 合成 128-bit）；改成两个 `float4` store 后 `S=1`
   **288us**。**做带宽下界必须显式 128-bit 读写**，否则量到的是编译器伪影。

### 3.2 三协议端到端：融合把 1.48× 提到 1.67×，顶到天花板

Part B。单步真实 GEMM：REF `k_grouped` fp32/累积 **499.0us（241 TFLOPS）**、
ALT `E×` 稠密 1D2D bf16 **219.6us（548 TFLOPS）**、**纯 GEMM 比 2.27×**（复现 P136）。
消费端单次 REF opt（`add_` fp32）458.0us。三协议端到端（`S` 扫 1→16）：

| S | REF e2e us | ALT-R e2e | ALT-OPT e2e | REF/ALT-R | REF/ALT-OPT |
|---:|---:|---:|---:|---:|---:|
| 1 | 957.0 | 1058.0 | **616.0** | 0.90× | 1.55× |
| 2 | 1455.9 | 1397.2 | 909.0 | 1.04× | 1.60× |
| 4 | 2453.9 | 1903.1 | 1489.7 | 1.29× | 1.65× |
| 8 | 4449.8 | 3051.2 | 2671.3 | 1.46× | 1.67× |
| 16 | 8441.5 | 5702.3 | 5045.4 | **1.48×** | **1.67×** |

```
REF e2e      = t_ref·S + opt_ref          (opt 读 fp32 梯度 + 读/写 master)
ALT-R e2e    = t_alt·S + torch.sum + opt_ref
ALT-OPT e2e  = t_alt·S + t_fused          (无 fp32 中间)
```

**三点结论**：

1. **`S=1` 时 ALT-R 反而更慢（0.90×）**——复现 P136 的 0.83×：要额外**物化一份 fp32 梯度**（470MB），
   而 GEMM 省的 280us 填不平这次物化 + 读回。**但只要融合（ALT-OPT），`S=1` 就是 1.55×**——
   没有 fp32 中间，少付的正是这笔账。
2. **`ALT-OPT` 稳定超过 `ALT-R`，且差距随 S 收敛到 ~13%**（`S=16`：1.67× vs 1.48×）。把两个协议的
   差别拆开：`ALT-R` 的 `torch.sum`（写 470MB fp32）+ 优化器读回（470MB）= **940MB/次**；
   `ALT-OPT` 的融合核把这两笔都省了，只保留「读 S 份 bf16 + 读/写 master」。
3. **`ALT-OPT` 顶到理论天花板**：`t_ref/(t_alt + 235MB/BW) = 499.0/(219.6 + 235e6/3.08e12·1e6) = 499.0/(219.6+76.3) ≈ 1.688×`，
   实测 `S=16` 的 **1.67×** 只差 ~1%。**天花板不是随 S 涨到 2.27×，而是卡在 1.69×**。

### 3.3 为什么 2.27× 不可达：那 76us/步的「读一遍」省不掉

把每步的账摊开（`S→∞` 极限，`t_alt=219.6us/步`）：

| 每步成本 | REF | ALT-OPT |
|---|---:|---:|
| GEMM（含自身写 bf16 235MB） | 499.0 | 219.6 |
| 消费端读 S 份 bf16 梯度（235MB） | — | **+76.3** |
| 优化器读/写 master（470+470MB，摊到每步 ~29us） | +28.6 | +19 |
| **合计/步** | **~528** | **~315** |
| **比值** | — | **~1.68×** |

**2.27× 只在「消费端读 S 份 bf16 梯度」这步成本为零时才成立**。可它必须发生：per-step 梯度是
**真实的 S 份 bf16 张量**（`ALT` 每步一份 fresh），不管谁来消费，都得把 `S·235MB` 读一遍。
把这份读按满带宽（3.08 TB/s）折算就是 **76.3us/步**，叠加在 219.6us/步的 GEMM 上——**这就是
1.69× 天花板的全部来源**。第 146 篇说的「只有优化器直接吃 bf16 才能回到 2.2×」**只对了一半**：
优化器直接吃能收回「物化税」（1.48→1.67），但收不回「读一遍」（1.69 的天花板）。

---

## 4. 结论：物化税可回收，2.27× 是「无消费成本」的虚构上界

| 量 | 值 | 含义 |
|---|---|---|
| 纯 GEMM 比（P136） | **2.27×** | 只算 GEMM，**隐含消费端读为 0** |
| `ALT-R`（bf16 + `torch.sum` + 优化器） | 1.46–1.48× | P136 的路线，付了物化税 |
| `ALT-OPT`（bf16 + 融合优化器） | **1.67×** | 收回物化税，≈顶到天花板 |
| 理论天花板 `t_ref/(t_alt+235MB/BW)` | **1.69×** | 每步梯度至少被读一遍 |

**给训练的落点**（与 P136 一致但更精确）：把 wgrad 输出换成 bf16 **并且**让优化器直接消费 bf16
（不物化 fp32 参数梯度），端到端能拿到 **~1.67×**——这是 SM90 上「不改 kernel、只改累积/消费协议」
能达到的实际上限。想更进一步，只能**减少「读一遍」的字节**：例如让优化器按 **fp32 master + bf16
梯度流式**消费（只读一次、不落中间），或把归约与优化器更新真正融进 GEMM 的 epilogue——但后者
在 SM90 上无对应 kernel（1D2D 只能 bf16 直写、1D1D 只能 fp32 累积，`sm90_fp8_gemm_1d2d.hpp:81` /
`sm90_fp8_gemm_1d1d.hpp:84`），仍是**协议改动**而非 kernel 微调。

**方法论副产品**：P136 的「归约税 838.6us」是 `torch.sum` 的数（本文复现 837.4us）；一个显式
128-bit 的手写核把它压到 762.6us（`S=8`）/ 1371.8us（`S=16`）。**报「归约税」时要声明实现**，
否则会把实现差距当物理天花板。

---

## 5. 与 SM100 的关系（仅分析）

SM100 的 wgrad 走 `k_grouped_fp8_gemm_tn_contiguous`（MN-major，`csrc/apis/gemm.hpp:320-371`），
`C` 可选但生产 FP8 NT 路径同样传 `C=D` 做 fp32 累积，`C/D` 是 3D `[G,M,N]` fp32（第 146 篇 §5）。
因此「每专家一份 fp32 + 回读」的结构在 SM100 上一致，本篇的**协议账（物化税 + 读一遍天花板）**
与架构无关。差别在算力/带宽比：SM100 的 HBM 带宽更高、UMMA 有 `cp.reduce.async.bulk` 等
更省的原语，1.69× 天花板的具体数值会变，但**「每步 bf16 梯度至少被消费端读一遍」这一项不可消**。
SM100 的 1D1D 直写/UMMA 细节 **仅分析**（本机 H100 跑不了）。

---

## 6. 边界

- **能跑/能核**：全部 SM90 FP8 kernel（H100 真机）；消费侧四个内核架构无关、直接编译运行真实设备代码，
  正确性对 CPU double（`max|err|=2.384e-07`）；计时 event 稳态（每配置 20 次 warmup + 10 次采样）。
- **不能核**：SM100 `tn` 路径仅源码分析；`k_grouped` 无 bf16 变体，ALT 用 `E×` 稠密 1D2D **模拟**
  「同一个 wgrad 数学」的等价写法，不是真正的 wgrad API（同 P136）。
- **消费端口径**：统一用 SGD-lite（`w ← w − lr·G`）；AdamW 会多读/写各自一份状态（fp32），
  但对三协议是**公共项**，不改变比值结构。`ALT-R` 用 `torch.sum`（P136 口径），另附手写核对照。
- **实验口径**：ALT 的 B 用 128×128 block SF、REF 的 A/B 均 1×128，故两条 GEMM 的量化噪声口径略有差异
  （P136 已讨论，不影响吞吐/协议账）。
- **环境**：单进程、空闲 GPU1（`CUDA_VISIBLE_DEVICES=1`）；8 卡节点共享，沿用本专题报数约定。
- **不变量**：`BW` 取实测消费侧 ~3.08 TB/s（纯读峰值 3009 GB/s）；若换更差的实现，天花板只会更低。

---

## 7. 复现

```bash
bash harness/exp_wgrad_reduce_consumer.sh > harness/exp_wgrad_reduce_consumer.out.txt
```

脚本先 `nvcc -arch=sm_90` 编 `src/wgrad_reduce_consumer.cu`（独立可执行跑 [A]/[B] 自带报告；
并编成 `.so` 由 Python 经 ctypes 驱动其消费侧内核），再跑 `harness/exp_wgrad_reduce_consumer.py`
（真实 SM90 GEMM + 三协议端到端）。

### 源码坐标（`~/github/DeepGEMM@057ca59`，行号已核对）

- `csrc/apis/gemm.hpp:373-427`：SM90 wgrad `k_grouped_fp8_gemm_nt_contiguous`（`:384` recipe、
  `:403` `c.has_value() and c.value().is_contiguous()`）。
- `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`：`c.has_value() and d==float`（1D1D 必 fp32 累积）。
- `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:81`：`not c.has_value() and d==bf16`（1D2D 必 bf16 直写）。
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`：`SM90_TMA_REDUCE_ADD_2D`（累积的落点）。
- `deep_gemm/utils/math.py:26/52/62`：`per_token_cast_to_fp8` / `per_channel_cast_to_fp8` /
  `per_block_cast_to_fp8`（两条路径的量化入口）。
- `csrc/apis/gemm.hpp:320-371`：SM100 wgrad `tn`（仅分析）。

### 实验

- `src/wgrad_reduce_consumer.cu`（`k_reduce8`/`k_fused_sgd8`/`k_reduce_atomic`/`k_read8` + 自测 + ctypes API）。
- `harness/exp_wgrad_reduce_consumer.py`（Part A 内核对照、Part B 三协议端到端）。
- `harness/exp_wgrad_reduce_consumer.sh`（driver，编 `.cu` 与 `.so`）。
- `harness/exp_wgrad_reduce_consumer.out.txt`（原始输出；`.cu` sha256 `3e65de90…`）。

### 线上资料

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM
- Mega MoE PR #316：https://github.com/deepseek-ai/DeepGEMM/pull/316
- 本专题：`docs/146`（wgrad 换 bf16 输出的精度代价）、`docs/145`（wgrad 强制累积代价量化）、
  `docs/12`（wgrad API）、`docs/15`（反向 ncu 归因）。
