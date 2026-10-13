# TileLang 基线 SwiGLU 算子：Mega MoE L1 epilogue 的官方逐位对照物

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P153-1 增量（第 163 篇）**。
> 前面第 03/09/10 篇拆了 SM100 主 kernel 的 MMA 与 L1 epilogue（in-place SwiGLU + 乘 top-k
> 权重 + UE8M0 量化），第 27 篇拆了参考量化器 `deep_gemm/utils/math.py`，第 41 篇拆了
> `tests/test_mega_moe.py` 的端到端正确性判据（`torch.equal` 逐位）。但那份测试的 **baseline**
> （`run_baseline`：DeepEP dispatch → grouped GEMM → **TileLang SwiGLU** → grouped GEMM → combine）
> 里真正承担「SwiGLU + 乘权重 + FP8 量化」这一步的算子
> `third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py` 一直只被顺带引用，从未单独读过。
> 本篇把它当 **fused mega kernel L1 epilogue 的可执行对照物** 拆透，并在 H100 上跑真实 kernel、
> 与独立 torch oracle 和「mega 设备端算术的 sm90 复刻」逐位对拍。
>
> **Mega MoE 融合 kernel 是 SM100 专属、本机跑不了**；但本算子（TileLang，编译成 CUDA）
> 与 mega epilogue 的算术/量化语义**架构无关**，H100(sm90) 上真机可跑。源码坐标：
> `~/github/DeepGEMM` commit `057ca59`（`#462`）。实验：
> `harness/exp_tilelang_swiglu.{py,sh}` + `src/swiglu_mega_epilogue.cu` →
> `harness/exp_tilelang_swiglu.out.txt`。

---

## 1. 它在哪里被用：baseline 的 L1 → L2 之间

`tests/test_mega_moe.py` 用**一次 launch 比两套实现**：`run_fused`（单个 SM100 mega-kernel）
对比 `run_baseline`（分步实现），并用 `torch.equal` 逐位验收（第 41 篇）。baseline 的三段是：

- 分派：`ep_buffer.dispatch(...)` 得到 `recv_x` / `recv_topk_weights`；
- **L1 grouped GEMM** → `l1_y`（`[num_recv_tokens, 2I]`，gate 与 up 拼接）；
- **本算子**：`tilelang_ops.swiglu_apply_weight_to_fp8(l1_y, recv_topk_weights, ...)`（`tests/test_mega_moe.py:319-325`）
  —— 一份算子同时做完 **SwiGLU + 乘 top-k 权重 + per-32 UE8M0 量化**，输出给 L2；
- L2 grouped GEMM → `combine`。

按 mma 类型分两套配置（`:277-302`）：

| 路径 | `round_scale` | `ue8m0_scale` | `output_bf16` | `num_per_channels` |
|---|---|---|---|---|
| FP8/FP4 路由（`fp8xfp8` / `fp8xfp4`） | `True` | `True` | `False` | 32 |
| BF16 路由（`bf16xbf16`） | `False` | `False` | `True` | 128 |

共享专家分支同样调它（`:285-298`，FP8 走 `num_per_channels=32`、BF16 走 128）。因此这个算子的
输出必须与 fused kernel 的 L1 epilogue **数值一致到逐位**——这就是为什么它值得单独拆。

## 2. host 包装（`:150-213`）

入口 `swiglu_apply_weight_to_fp8` 是一个薄 host 壳：

- **契约**：`assert fmt == "e4m3"`（`:168`，目前只支持 E4M3）；`assert hidden_size % (2*num_per_channels) == 0`
  （`:169`，`hidden_size` 是拼接后的 `2I`）。
- **`num_sms` 默认取 `deep_gemm.get_num_sms()`**（`:170-171`）——注意这个数会**进 JIT 模板常量**
  （见 §3 的 `num_ctas`），所以它是编译期参数。
- **两个输出容器**：`y`（`[num_tokens, I]`，`float8_e4m3fn`，`:173-177`）与 `y_sf`
  （形状由 `get_sf_shape` 定，dtype `uint8`（UE8M0）或 `float32`，`:178-182`）；`output_bf16`
  时额外给 `y_bf16`（`:182`）。
- **`pass_configs`（`:185-189`）**——三个 TileLang 编译开关：`TL_DISABLE_WARP_SPECIALIZED: True`、
  `TL_DISABLE_TMA_LOWER: True`（关掉 warp-specialized + TMA lowering，退回普通 parallel-loop 代码）、
  `TL_ENABLE_FAST_MATH: fast_math`（默认 `True`，对应 `T.exp` 快数学）。
- **发射（`:190-204`）**：11 个编译期参数（`hidden_size//2`、`num_per_channels`、两个 bool、
  `num_sms`、四个 `is not None`/选项推出）全部进 kernel 特化；运行时喂 `x / topk_weights /
  avail_tokens.view(1) / y / y_sf / y_bf16 / clamp_value`。
- **收尾（`:206-212`）**：UE8M0 时把 `y_sf` 视图成 `int32`（4 个 UE8M0 打进一个 int32），
  再 `.T[:num_tokens]` 转成 **token-major** `[num_tokens, ceil(ceil(I/g)/4)]`。

## 3. 设备 kernel（`:10-147`）

`@tilelang.jit` 装饰的 `_swiglu_apply_weight_to_fp8_tl` 返回一个 `@T.prim_func`。

**几何（`:30-40`）**：`assert half_hidden % 16 == 0`；`num_block_h = max(half_hidden // gcd(half_hidden, 16*1024), 1)`
（把 hidden 切成至多 `num_sms` 份，每份 `blk_h` 列）；`layout_h = blk_h // 16` 是每 1024 线程一行
放多少 token；`blk_n = 1024 // layout_h`。每个 CTA 用 1024 线程、处理 `blk_n` 个 token × `blk_h` 列的一个 tile。

**persistent 调度（`:136-144`）**：

```
with T.Kernel(num_ctas, threads=1024) as cta_id:
    avail_tokens_l = avail_tokens[0] if has_avail_tokens else num_tokens
    T.pdl_sync()                       # ：138，avail_tokens 必须是常量（等前序 kernel）
    if cta_id >= new_num_ctas * num_block_h: T.thread_return()   # ：143，多余 CTA 早退
    for bi in T.serial(cta_id // num_block_h, avail_tokens_l - thread_idx // layout_h * new_num_ctas, new_num_ctas * blk_n):
```

即「一列块一个 CTA、跨 token 网格步进」，上界减去 `thread_idx // layout_h * new_num_ctas` 做**尾裁剪**
（保证不越界读写）。`T.pdl_sync()`（`:138`）对应 host 侧 `do_cpu_sync=False` 时 `avail_tokens` 由
前序 kernel 写入（第 51/52 篇的 PDL 依赖）。

**计算（`:78-121`，`@T.macro main`）**：

- **取操作数**：`gate_frag = x[:, :blk_h]`、`up_frag = x[:, half_hidden + ...]`（`:78-82`）——
  **gate 在前半、up 在后半**；
- **权重**：`topk_weight[i] = topk_weights[...] if has_topk_weights else 1.0`（`:84-85`）——
  权重在 L1 epilogue 就乘进激活（与 mega kernel `:1134` 一致）；
- **clamp（`:88-91`）**：`up = min(c, max(-c, up))`；**`gate = min(c, gate)`，没有下界**；
- **SwiGLU（`:92-95`）**：`y = gate/(1+exp(-gate)) * up * weight + 0`（末尾 `+ 0` 是注释里的
  "HACK : + 0 for vectorize"，为了向量化）；
- **amax（`:98-100`）**：`T.reduce_absmax` 沿每个 `num_per_channels` 连续组取 `|y|` 最大，
  再 `clamped_amax = max(amax, 1e-4)`；
- **SF（`:101`）**：`get_sf_and_inv(clamped_amax, round_scale, ue8m0_scale)`；
- **存 SF（`:104-109`）**：UE8M0 时 `out_sf[scale_idx//4, token*4 + scale_idx%4] = sf`（4 个指数打进
  一个 word 的 4 字节）；列主序非 UE8M0 时 `out_sf[scale_idx, token]`；否则 `out_sf[token, scale_idx]`；
- **量化（`:113-117`）**：`y_f8 = y * sf_inv`（幂次乘法精确）→ 存 E4M3；
- **BF16 旁路（`:119-121`）**：`output_bf16` 时另存量化前的 `y_frag` 为 bf16。

## 4. host 公式 `utils.py`：两个不平凡点

`get_sf_shape`（`utils.py:13-30`）：

```
num_scales = ceil_div(hidden, num_per_channels)          # ：20
num_scales = ceil_div(num_scales, 4) if use_ue8m0 else num_scales   # ：21  ← UE8M0 每 4 个一包
num_sf_tokens = num_tokens
if use_col_major_sf:
    num_sf_tokens = align(num_tokens, 4)                 # ：27  ← TMA 16B 对齐（4 字节）
    num_sf_tokens = num_sf_tokens * 4 if use_ue8m0 else num_sf_tokens  # ：28  ← 每 token 展开 4 字节
return (num_scales, num_sf_tokens) if use_col_major_sf else (num_sf_tokens, num_scales)  # ：30
```

两个点：① UE8M0 把每 4 个指数打包成一个 word，故 scale 维要 `ceil/4`；② **列主序 SF 必须 TMA 对齐**，
token 维 `align(4)` 且 UE8M0 时再 `×4`（每个 token 占 4 字节——这正是 §2 末尾 `.view(int32).T[:T]`
能还原出 `[T, words]` 的原因）。实测 5 组形状与独立公式逐点一致（§6 A）。

`get_sf_and_inv`（`utils.py:33-47`）就是 **ceil-to-power-of-two**：

```
sf = amax / 448.0                          # ：34  （448 = E4M3 max finite）
if not round_sf: return sf, 448.0/amax     # ：35-36  非幂次路径（BF16 用）
bits = T.reinterpret("uint32", sf)
exp = (bits >> 23) & 0xFF                  # ：40
man_bits = bits & ((1<<23)-1)              # ：41
exp_scale = T.reinterpret("int32", exp - 127 + (man_bits != 0))   # ：42  ← 尾数非 0 则进位
if use_ue8m0: sf = T.Cast("uint8", exp_scale + 127)               # ：44  ← UE8M0 指数
else:         sf = T.reinterpret("float", (127 + exp_scale) << 23)  # ：46
return sf, T.reinterpret("float", (127 - exp_scale) << 23)        # ：47  ← sf_inv，幂次精确
```

**`exp_scale = exp - 127 + (man_bits != 0)`** 就是「把 `amax/448` 向上取到 2 的幂」：尾数非 0 时指数 +1。
这与 mega kernel 用的 `get_ue8m0_sf_exp`（`common/math.cuh:111-135`，**用整数加法的进位**实现同一
取整，见第 27/44 篇）逻辑等价；`sf_inv = 2^(127-exp_scale)` 是幂次，乘积精确为 1。实测 SF 与
`deep_gemm.utils.math.ceil_to_ue8m0(amax/448)` 的指数**逐点 0 不符**（§6 A）。

## 5. 与 SM100 mega kernel L1 epilogue 的对照

mega 主 kernel 的 L1 分支（`sm100_fp8_fp4_mega_moe.cuh:1066-1205`）与本算子**同一数学**，
差异只在「数据从哪来」和「用什么指令」：

| 环节 | TileLang 基线（本算子） | SM100 mega kernel |
|---|---|---|
| gate/up 来源 | `l1_y` 前半/后半（`:78-82`） | TMEM 里 **gran-8 交错**的 gate/up（权重已 `transform_weights_for_mega_moe`，第 02 篇） |
| bf16 round | 无需（输入已是 bf16） | `__float22bfloat162_rn`（`:1114-1115`） |
| clamp | `up=clamp(-c,c)`、`gate=min(c,·)`（`:89-91`） | `__hmin2/__hmax2`，**同为** `up` 双侧、`gate` 单侧（`:1116-1119`） |
| SwiGLU | `gate/(1+T.exp(-gate))`（`:92-94`） | `__expf(-gate)` + `math::fast_rcp`（`:1124-1132`，`kFastMath`） |
| 乘权重 | `* topk_weight`（`:92-94`） | `* weights`（`:1134`，从 `l1_topk_weights_buffer` 每次 32 token 读） |
| amax | `T.reduce_absmax` 每 `num_per_channels` 组（`:98`） | 线程→warp→warp-pair 归约（`:1136-1171`） |
| SF | `get_sf_and_inv(ceil 幂次)`（`:101`） | `math::get_ue8m0_sf_exp/inv`（`:1173-1176`） |
| 量化 | `y * sf_inv` → E4M3（`:113-117`） | `__fmul2_rn(y, sf_inv)` → `__nv_fp8x4_e4m3`（`:1183-1186`） |
| SF 落地 | `out_sf[scale//4, token*4+scale%4]`（`:106`） | `l2_sf_buffer` MN-major packed UE8M0（`:1190-1211`） |

关键结论：**两者的 silu 用了不同的近似**（`T.exp`+除法 vs `__expf`+`rcp.approx`），但因为之后要
E4M3 量化（相对精度 ~2⁻³），差异在绝大多数元素上被舍入归掉——这正是 §6 D 要量的事。

## 6. H100 真机实测

全部在 `kernel_lab`（H100/sm90）真跑；真实源码 sha256 见 `.out.txt`
（`swiglu_apply_weight_to_fp8.py 301783c3…`、`utils.py aab1e230…`、`common/math.cuh 576e9db1…`）。

### A) host 公式
5 组 `get_sf_shape` 与独立公式**逐点一致**（含 `T=65` 对齐到 272、`g=128`、`T=1`）；
`get_sf_and_inv` 产出的 SF 与 `ceil_to_ue8m0(amax/448)` 的指数 **mismatch=0/2048**；
UE8M0 样本值 `[121,120,120,119,121,120]`（即 2⁻⁶ 量级）。**ALL PASS**。

### B/C) 真实 tilelang kernel vs 独立 torch oracle
`T=512, 2I=4096`，五种配置全部逐位对拍：

| 配置 | FP8 mismatch | SF mismatch | 量化 relerr |
|---|---|---|---|
| FP8 g=32 | **0 / 1,048,576** | **0 / 32,768** | 2.253e-2 |
| FP8 g=128 | **0 / 1,048,576** | **0 / 8,192** | 2.253e-2 |
| FP8 g=32 + clamp=8 | **0 / 1,048,576** | **0 / 32,768** | 2.253e-2 |
| FP8 g=32 无权重 | **0 / 1,048,576** | **0 / 32,768** | 2.253e-2 |
| BF16 g=128 | bf16-out mismatch **3 / 1,048,576** | — | max\|Δ\|=1.95e-3 |

50 万个 FP8 元素 + 全部 SF **逐位等于**「按文档语义重写的 torch oracle」——坐实本算子的
SwiGLU/clamp/权重/量化语义与 mega epilogue 的**数学**完全一致，量化 relerr 2.25e-2 就是纯
E4M3 噪声（与第 89 篇前向复刻量到的同量级）。BF16 分支只有 **3 个元素差 1 个 bf16 ulp**
（`T.exp` 快数学 vs torch sigmoid 的舍入），在 `torch.equal` 验收里由后续 L2/combine 的 bf16
舍入吸收。

### D) mega L1 epilogue 算术（真实设备头）vs tilelang 逐位
`src/swiglu_mega_epilogue.cu` 用真实 `common/math.cuh` 的 `get_ue8m0_sf_exp/inv`、`fast_rcp`，
把 mega epilogue 的算术搬到 sm90（`T=256, I=2048, g=32`）：

| 变体 | FP8 mismatch | SF mismatch |
|---|---|---|
| `fast=1`（`__expf`+`fast_rcp`） | **0 / 524,288** | **0 / 16,384** |
| `fast=0`（`expf`+除法） | **0 / 524,288** | **0 / 16,384** |

**两者都与 tilelang 逐位相等**：mega 的 `__expf`/`fast_rcp` 近似在最坏情况下也不改变 E4M3 字节，
坐实「baseline 的 tilelang SwiGLU 与 fused kernel 的 L1 段 **bit-compatible**」——这正是官方
`torch.equal` 验收能成立的前提。

### E) 吞吐
`T=8192, I=2048, g=32`：**122.2 us/iter、686 GB/s**（读 `[8192,4096]` bf16、写 `[8192,2048]` fp8，
共 8.4×10⁷ 字节）。该算子本身是**带宽受限**的元素级融合——语义上等价于一次「读 L1 输出、写 L2 输入」，
mega kernel 把这一步折进 L1 epilogue（省掉这次 global 往返），是它融合收益的一部分。

## 7. 结论与边界

- **定位**：`swiglu_apply_weight_to_fp8.py` 是 Mega MoE 官方 baseline 的 SwiGLU 段，也是 fused
  kernel L1 epilogue 的逐位对照物；host 公式 `get_sf_shape`/`get_sf_and_inv` 就是 UE8M0 打包与
  ceil-to-power2。
- **语义**：clamp（`up` 双侧、`gate` 单侧）→ `silu(gate)*up*weight` → per-`num_per_channels`
  amax（≥1e-4）→ ceil 幂次 SF → E4M3。与 mega kernel `:1066-1205` 同一数学。
- **实测**：50 万 FP8 元素 + 全部 SF（含 g=32/128、clamp、无权重）**逐位等于**独立 oracle；
  mega 设备端算术（含 `__expf`+`fast_rcp`）与 tilelang **逐位相等**；吞吐 686 GB/s。
- **边界**：① 融合 kernel 本体 SM100 专属、仅分析；② 本算子只支持 E4M3（`assert fmt=="e4m3"`）、
  非 UE8M0 路径未经本实验覆盖；③ BF16 分支有 ~3/10⁶ 的 1-ulp 差异（快数学）。

（第 41 篇拆了这份 baseline 所在的端到端测试、第 89 篇用 SM90 grouped GEMM 复刻了整条前向流水线；
本篇补上了其中一直缺席的 SwiGLU 算子本体。）
