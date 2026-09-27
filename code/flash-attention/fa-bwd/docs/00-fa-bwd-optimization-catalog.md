# FlashAttention 反向优化手段梳理（FA / TE 对照）

> 目标：为「在目标 AI 卡上开发 flash-attention-backward」做技术摸底。本文梳理 FA2/FA3 反向的
> 实现结构与优化手段，以及 FP8 反向（TE 做法）的关键难点。**行号引用基于本地
> `~/github/flash-attention` commit `edb5c76`**，标注为 `文件:行`，需在写作/实现时回源码复核。
> 后续实现与实测见 `../ROADMAP.md`。

---

## 1. 反向的数学与计算量

设 `S = scale·QKᵀ`，`P = softmax(S)`（causal），`O = PV`，损失对 `dO` 已知。反向：

$$
\begin{aligned}
D &= \operatorname{rowsum}(dO \odot O) \quad(\text{即 }\sum_j dO_{ij}O_{ij})\\
dP &= dO\,V^{\mathsf T}\\
dS &= P \odot (dP - D)\\
dV &= P^{\mathsf T} dO\\
dQ &= \text{scale}\cdot dS\,K\\
dK &= \text{scale}\cdot dS^{\mathsf T} Q
\end{aligned}
$$

要点：
- **`D`（delta / softmax_d）可以预计算**，与 `P` 无关（只需 `dO,O`）。FA 用独立 preprocess kernel 算它，
  这样主 kernel 不必同时持有 O 和 dO（省 smem/寄存器）。
- `dQ` 需要对**所有 K 块**求和（沿列归约），`dK,dV` 对**所有 Q 块**求和（沿行归约）。
- 反向 FLOPs ≈ 前向的 2×：约 `4·b·s²·h·(qk_dim+v_dim)`。
- 反向要读 `Q,K,V,O,dO` 写 `dQ,dK,dV`；若像前向一样把 `P` 存下来，显存会是 `O(N²)`——所以必须
  **recompute**。

---

## 2. FA2（SM80）反向结构

核心文件：`csrc/flash_attn/src/flash_bwd_kernel.h`、`flash_bwd_preprocess_kernel.h`、
`flash_bwd_launch_template.h`。

### 2.1 三段式

1. **preprocess**（`flash_bwd_preprocess_kernel.h`）
   - `compute_dot_do_o`（`:58`）：逐元素 `dot(dO,O)` 行和 → `softmax_d`（即 `D`），并做 NaN 检查。
   - `clear_dKVaccum`（`:145`）：清空 `dK/dV` 的累加缓冲。
   - `convert_dQ`（`:185`）/ `convert_dKV`（`:275`）：把 fp32 累加缓冲转回目标精度写回。
2. **main kernel** `compute_dq_dk_dv`（`flash_bwd_kernel.h:799`）
   - 调用 `compute_dq_dk_dv_1colblock`（`:81`）逐 K/V 列块处理。
   - causal 用 `Is_first/Is_last` 模板区分「对角块（需 mask）」与「全块（无需 mask）」（`:813-820`）。
   - 另有 `compute_dq_dk_dv_seqk_parallel`（`:826`）做序列并行（带 `Seq_parallel`）。

### 2.2 数据流（每个 K/V 列块，`1colblock`）

对 Q 块 `[BLK_M, d]` 与 K/V 块 `[BLK_N, d]`：
1. 从 smem 取 Q,K，算 `S = QKᵀ`（`P` 用 online-softmax 的前向 `m` 重新 exp 得到，**recompute**）；
2. `dV += Pᵀ dO`（对 N 累加）；
3. `dP = dO Vᵀ`；
4. `dS = P∘(dP − D)`；
5. `dQ += dS K`（Q 方向在寄存器/smem 累加，最终写 `dQ_accum`）；
6. `dK += dSᵀ Q`。

### 2.3 优化手段清单（FA2）

| 手段 | 作用 | 证据 |
|---|---|---|
| **recompute P** | 不存 `O(N²)` 的 P，反向用 `m` 重新 exp | `flash_bwd_kernel.h` 主循环 |
| **预计算 D=rowsum(dO∘O)** | 主 kernel 不必同时驻留 O/dO | `flash_bwd_preprocess_kernel.h:58` |
| **dQ 累加缓冲 `dQ_accum`** | dQ 跨 K 块累加，主 kernel 用 `atomicAdd` 写缓冲 | `flash_bwd_kernel.h:124-125, :678` |
| **dK/dV 原子累加** | 跨 Q 块累加；非确定性模式直接 `atomicAdd` | `:678` 附近 |
| **deterministic 模式** | 每 Q 块写独立 `dQ_accum` split，再 convert，结果可复现 | `:124-125`, `:826` |
| **causal 模板特化** | 对角块 mask、其余全块，省无效计算 | `:813-820` |
| **smem 复用 Q/K/V 分块** | 减少全局访存 | Kernel_traits 分块 |
| **warp 划分（N 方向切分）** | 4 warps 各算 S 的一段，dK/dV 后合并 | Kernel_traits |
| **fp32 累加** | S/dS/dQ/dK/dV 均 fp32 累加，降低误差 | acc_dq/acc_dkv |

---

## 3. FA3（SM90 Hopper）反向结构

核心文件：`hopper/flash_bwd_kernel_sm90.h`、`hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp`、
`hopper/epilogue_bwd.hpp`、`hopper/tile_scheduler.hpp`。

在 FA2 基础上叠加 Hopper 特性：
- **TMA**（`cp.async.bulk.tensor`）搬 Q/K/V/dO 分块，配 `mbarrier`；
- **warpgroup MMA（wgmma）** + **warp specialization**（producer 搬数、consumer 算）；
- **ping-pong / 多级流水**（`sm90_pipeline_no_cluster.hpp`）；
- `TiledMmadKV`、`dKV_swapAB`（`flash_bwd_kernel_sm90.h:39-44`）——dK/dV 的 MMA 布局交换以适配 wgmma；
- epilogue 单独在 `epilogue_bwd.hpp` 里 store dK/dV（`:268`）。

> 论文结论（需实测复核）：FA3 反向相对 FA2 主要赢在 **TMA + wgmma + 更深的流水**，
> 而不是算法变化。

---

## 4. FP8 反向：为什么最难点，TE 怎么做

FA 仓库的**反向没有 FP8**（`csrc/flash_attn/src` 只有 fp16/bf16 的 `flash_bwd_hdim*`；
`grep -ril fp8` 命中的是前向/interface）。FP8 反向要借鉴 **TransformerEngine fused_attn_bwd**。

### 4.1 TE 的 FP8 反向口径（来自 `te-perf/bench_te.py` 与 TE 文档）

- 前向：Q/K/V/S/O 用 **E4M3**；
- 反向：`dO`、`dP`（以及 `dQ/dK/dV`）用 **E5M2**（动态范围大，适合梯度）；
- 输入张量在计时区外**预先量化**（rowwise，`Float8Quantizer(rowwise=True)`），
  所以测到的是纯 FP8 kernel；kernel 内部做 rowwise 动态缩放。
- 反向支持受 head_dim 限制：训练口径 `qk==v` 最大 256，`qk!=v` 仅在 `qk≤192,v≤128` 附近可用
  （见 `te-perf/results_summary.md`）。

### 4.2 关键难点

1. **量化对象**：`dO` 与 `P`/`dS` 都要进张量核。`P=softmax` 在 `[0,1]`，动态范围小，直接 E4M3；
   `dO` 无界，需 E5M2 + rowwise scale。`dS = P∘(dP−D)` 的量化会放大误差。
2. **缩放因子（scale）**：FP8 张量核要 `scale_a*scale_b` 乘回 fp32 累加器。rowwise（每行一个 amax）
   够用但精度一般；需与 TE 的 scaling 布局严格对齐才能数值可比。
3. **累加精度**：所有 MMA 累加器保持 **fp32**；只在输入侧量化。`D`、`m`、`l` 等统计量用 fp32。
4. **量化误差对 dQ/dK/dV 的影响**：反向对 S 的误差敏感（softmax 的 `dS` 含 `P` 因子，小 P 处噪声相对大），
   需要对拍确定容差（预期比 fp16 松 1~2 个数量级）。
5. **确定性**：FP8 kernel 一般非确定性（原子累加）；对拍用容差而非位相等。
   → **已提供 opt-in 的确定性模式 `--det`（P3-4e，第一百一十八轮）**：把 dK/dV 的跨 CTA
   `atomicAdd` 换成「按 (Q 头, Q 块) 分片的 partial 覆盖写 + 固定次序二次归约」，两次跑
   逐位相同（`runs[1-2] bitwise-diff = 0`）；代价是 ksplit 固定为 1 + 一次纯带宽 bound 的
   归约（S=4096 时 reduce 731.6µs、DRAM 91.5%），见 `docs/03` §56。
   → **`--detk=N>1`（P3-4f，第一百一十九轮）把 `--det` 扩到 split-K**：dK/dV 的 partial
   天然无需 part 维（一个 `(mblk,jg)` 只属于一个 part），只需给 dQ 加 part 分片 partial +
   `dq_reduce_kernel`，即可在 ksplit>1 下保持 dq/dk/dv 全逐位可复现；DET 相对 k=1 提速
   S512 **1.31×** / S4096 **1.11×**（k=4 触底）。见 `docs/03` §57。
   → **`--det` 已扩到 Hopper TMA 快路（P3-4g，第一百二十轮）**：Q/dO/K/V 全 4D-TMA 的
   `kvtma` 主 kernel 也能走 deterministic dK/dV（定长 HD=128/GQA、ksplit=1）；三 shape ×
   单/两文件两次跑逐位相同、与 atomic 差 e-7–e-6，代价仍是 reduce 的 DRAM partial 写读。
   见 `docs/03` §58。
   → **`--det` 已扩到 MLA（HD=512，P3-4h，第一百二十一轮）**：`dkv_reduce_kernel<HD,BM>` 与
   body 的 DET 分支本就 HD 无关，device 数学一行未改；MLA 的 dQ 无法用寄存器累加 ⇒ 锁
   ksplit=1 的单写者 `red_add2` 保证 dQ 确定。三 shape × 单/两文件两次跑
   `bitwise dq/dk/dv = 0`、`DET-vs-atomic` dq 恒 0 / dk,dv e-7–e-6；代价 S256H2 1.035×、
   S512H4 0.923×、S1024H2 0.940×，bound 仍是 reduce 的 DRAM 带宽。见 `docs/03` §59。
   → **`--det` 已扩到 varlen（P3-4i，第一百二十二轮）**：body 的 DET 分支对定长/变长通用，
   只需 host 传 `S=maxlen/nblk=nblk_max`；新增 `dkv_reduce_varlen_kernel<HD,BM>` 按
   `cu_seqlens` 的逐序列 `len_b/nblk_b` 定界、输出按 packed token 定位。4 个 case（MHA 单长/
   不齐/GQA q32kv8）× 单/两文件、ksplit=1/4 两次跑 `bitwise dq/dk/dv = 0`，
    `DET-vs-atomic` e-7–e-6；代价 0.71–0.91×，bound 仍是 reduce 的 DRAM 带宽（87.4%）。
    见 `docs/03` §60。
   → **`--det` 已扩到 MLA（HD=512）的 varlen（P3-4j，第一百二十三轮）**：device 一行未改，
    host 复用 `dkv_reduce_varlen_kernel<512,64>`（P3-4i 的 HD 参数化布局红利）；MLA 的 dQ
    不可寄存器累加 ⇒ 锁 ksplit=1（单写者 `red_add2`）。3 个 case（b1 causal/b3 causal/b1 full）
    × 单/两文件两次跑 `bitwise dq/dk/dv = 0`、`DET-vs-atomic` dq 恒 0 / dk,dv e-7–e-6；
     代价（只差 DET）0.95–1.02×，但**锁 k=1 相对 auto split 的主 kernel ~3.3–5.6×**（确定性 =
     用单写者换掉 split-K 并行度）；reduce **48.03µs / DRAM 65.8% / L2 67.6%**（纯带宽 bound）。
     见 `docs/03` §61。
   → **`--det` 的 MLA（HD=512）支持 `ksplit>1`（P3-4k，第一百二十四轮）**：非 `kRegDq` 的 dQ
   epilogue 在 `DET && ksplit>1` 时把目标从 `dq_acc` 换成按 part 分片的 `dq_part`
   （同 `(mblk,h,part)` 只被一个 CTA 写、CTA 内 nt 程序序 ⇒ 天然确定），再接已就位的
   `dq_reduce_kernel<512>` 固定次序求和；device 只动这一处、host 扩 A/B。3 个定长 shape
   × 单/两文件、ksplit=1/4/8/16 **`runs[1-2] bitwise dq/dk/dv = 0`**；DET 在 **k=4 触底**、
   相对旧「锁 k=1」主链 **2.4–2.8×**（S1024H2 0.7256→0.2589ms），仍比 atomic 慢
   0.61–0.73×；`DET-vs-atomic` k=1 时 dq 恒 0、k>1 时 e-7。bound = `dkv_reduce_kernel` 的
   纯 DRAM 带宽（29.6µs/DRAM 77%/L2 76%）。见 `docs/03` §62。
   → **`--det` 的 MLA（HD=512）varlen 也支持 `ksplit>1`（P3-4l，第一百二十五轮）**：P3-4j 的
   A/B 仍锁 `ksplit=1`；本轮观察与 P3-4k 同构（varlen body 的 DET 分支与定长无关、dQ 的
   per-part partial 用 packed 全局 token），**device 一行未改**，host 只补 `dq_part` 分配/清零
   + `dq_reduce_kernel<512><<<(T,H),512>>>`。b1_t512/b3_t1792 ksplit=1/4/8/16、b1 full k=4
   × 单/两文件全部 `runs[1-2] bitwise dq/dk/dv = 0`；**DET 在 k=8 触底**、相对「锁 k=1」
   b1 **3.69×**/b3 **2.13×**；ncu `dkv_reduce_varlen_kernel<512,64>` 48.13µs/DRAM 65.7%、
    `dq_reduce_kernel<512>` 23.33µs/DRAM 81.7% ⇒ bound = reduce 的纯 DRAM 带宽。
    见 `docs/03` §63。
   → **`--det` 接进 MLA 的 K/V `cp.async` 回填流水（kvpipe，P3-4m，第一百二十六轮）**：
   此前 host 的 `launch_bwd_main_det` 写死 `KVPIPE=false`，MLA DET 一直走非 kvpipe 主 kernel；
   本轮加 `bool KVPIPE` 模板参、`kSmem` 选 `smem_bytes_kvpipe`（MLA 229888B ≤ 232448），
   P3-4k/P3-4l 的 A/B 各加一个 kvpipe DET 变体。**kvpipe DET 与非 kvpipe DET 的 dq/dk/dv
   逐位相同**（`runs[1-2]=0`、`kvpipe-vs-非kvpipe=0`）；DET 非 kvpipe→kvpipe 5 shape 1.018–1.090×
   （单文件 b3 1.095×）；ncu 主 kernel 243→220µs、`long_scoreboard` 3.93→3.25（机制同 O51）。
   见 `docs/03` §64。
   → **把两个二次归约融合进一个 kernel（P3-4n，第一百二十七轮）**：DET 的二次归约原本是两次
   launch（`dkv_reduce_kernel` 读 dk/dv partial + `dq_reduce_kernel` 读 dq partial）。新增
   `dkv_dq_reduce_kernel<HD,BM>` / `dkv_dq_reduce_varlen_kernel<HD,BM>`，grid =
   `dkv_blocks + dq_blocks`（1D，前 `dkv_blocks` 做 dK/dV、其余做 dQ），求和次序逐字沿用两个
   旧 kernel ⇒ **融合版与分开版 dq/dk/dv 逐位相同**。字节数不变（两 reduce 读各自 partial），
   收益来自「省一次 launch + 用轻量 dq 块（ncu DRAM 58.7%、8.6µs 尾）填满重 dkv 块的发射口」：
   reduce-only S512 **1.149×**、varlen D128 b1_t512 **1.170×**、S4096 **1.006×**（纯带宽墙、中性）；
   ncu 融合 22.94µs/DRAM 79.7%/occ 86.1%（分开 dkv 17.31µs/70.4% + dq 8.64µs/58.7%，合计
   25.95µs）。`--nofusered` 退回两次 launch。见 `docs/03` §65。
   → **varlen DET partial 试换 compact per-sequence 布局（P3-4o，第一百二十八轮）**：
   `fp8_mma_body` 的两个 DET 写点 + `dkv_reduce_varlen_kernel` / `dkv_dq_reduce_varlen_kernel`
   加行前缀和 `part_base`，非空时紧凑编址，空时逐字退化。**逐位相同**（只改地址）；缓冲大幅
   缩小（b4_t3840 D128 2147→713MB、b8_t2904 4295→575MB、D512 b3 201→88MB），但**耗时
   0.944–0.975×（负结果）**——reduce 只读被写过的条目，stride 空洞不产生 DRAM 流量；两布局都
   DRAM/L2 bound（86.9% vs 84.4%）。默认保持旧布局，`--partcompact` opt-in 只为省显存。真正的
   「减 partial 字节」只能靠 BM=128 跨 warpgroup 偏和（fp8 撞 smem 硬墙）。见 `docs/03` §66。
   → **partial 降精度存储（O60，第一百二十九轮，fp16）**：把 `--det` 的 dK/dV partial 从 fp32
   降到 fp16（`__half2`）。**reduce 单向 1.27–1.66×**（纯 DRAM 读字节减半），**但 DET 主 kernel
   写侧慢 0.84–0.87×**——ncu 实测 fp16 把 DRAM 写字节减半、**store 扇区数却一字不变**（DET
   partial 写是扇区粒度 bound，每 `(j,row)` 仅 16B 落不满 32B 扇区），故端到端只中性偏负
   （S4096 1.006×、S512/GQA 0.93–0.97×）。确定性保留、只改数值口径（~1e-3）。见 `docs/01` §17。
   → **确定性反向 `--det` 补齐 bf16（O61，第一百三十轮）**：把 fp16 的 O7b（partial + 固定次序
   二次归约）+ O60（partial 降精度存 fp16→bf16）逐字 dtype 参数化到 bf16（此前 bf16 无 `--det`），
   **三 dtype（fp16/bf16/fp8）现均有确定性反向**。两档 partial `runs[1-2]=0`；bf16-vs-atomic 仅
   bf16 舍入（7.8e-3–2.2e-2），默认路径逐位不变。性能与 fp16 O60 逐项同构：**reduce 单向
   1.27–1.66×（纯 DRAM 读字节减半），DET 主 kernel 写侧 0.84–0.87×**——bf16 把 DRAM 写字节
   减半但 **`st.global` store 扇区数几乎不变**（35.65M→35.54M），partial 写是扇区粒度 bound，
   端到端 `--det` 仍中性偏负。见 `docs/01b` §6at。
   → **partial 写扇区化（O62，第一百三十一轮，fp16+bf16；确定性 `--det` 端到端转正）**：
   O60/O61 的病根是「每 lane 只写 4B half2，quad 16B 落不满 32B 扇区」。O62 把**相邻两个列组
   `j`/`j+1` 的 4B 拼成一次 8B 写**（`uint2`），quad 4 lane 覆盖连续 32B；为让每 lane 的 8B
   恰好是这两段，partial 的 HD 维做**16 列块内置换**（`dkv_p16_perm`，reduce 端按同一置换读回
   ⇒ 求和集合/次序不变、数值逐位不变）。**store 扇区数减半**（fp16 S4096 35.65M→18.35M），
   DET 主 kernel 写侧从 0.84–0.87× **翻正为 1.10–1.17×**，reduce 仍 1.24–1.66×，**端到端
   `--det` 首次全面快于非确定性 atomic**（fp16 DET(fp16)/atomic = 1.19×/1.09×/1.03×；
   bf16 1.19×/1.10×/1.02×）。见 `docs/01` §18、`docs/01b` §6au。

### 4.3 我们的 FP8 反向实现路线（计划）

- 以 FA2 的 `1colblock` 结构为骨架，把 `dO`/`V`/`P`/`Q`/`K` 按 rowwise 量化到 FP8，
  用 `mma` FP8（`m16n8k32` e5m2/e4m3）做 `dP=dO·Vᵀ`、`dV=Pᵀ·dO`、`dQ=dS·K`、`dK=dSᵀ·Q`；
- `D`/`dS` 计算和 `dS` 量化在 fp32 完成后再量化；
- 先做 **正确性**（对 fp32 ref 的容差），再对标 TE fp8 的时间。

---

## 5. 目标 AI 卡的可移植性注意点

（当前开发机为 H100 sm90，反编译/运行均在此验证；目标卡待定，故代码尽量抽象。）

- 把「张量核 MMA 指令」「异步拷贝」「warp 调度」抽成可替换层，便于换卡；
- 先写**功能正确的标量/CUDA-core 版本**作为 golden，再逐层上张量核；
- 数值对拍统一走 `harness/fa_bwd_bench.py` dump 的 CPU npy，跨卡可比。

---

## 6. 待办

见 `../ROADMAP.md`：按 fp16→bf16→fp8 的顺序，各做「单文件 / 两文件」实现 + 编译 + ncu + 对拍 + 对标 TE，
并持续沉淀到本目录 `docs/`。
