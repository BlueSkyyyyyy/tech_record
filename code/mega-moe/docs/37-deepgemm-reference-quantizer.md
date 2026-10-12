# DeepGEMM 的测试参考量化器：FP8/FP4 block-scale 的精确语义

> 本文对应路线图 **P27-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 37 篇。
> 目标：把 DeepGEMM 里**贯穿所有 FP8/FP4 kernel 的量化语义**——`deep_gemm/utils/math.py` 的
> `per_token_cast_to_fp8` / `per_block_cast_to_fp8` / `per_token_cast_to_fp4` / `ceil_to_ue8m0` /
> `cast_back_from_*`——单独讲透。这些函数**定义了「量化后的操作数是什么」**，第 25/26 篇（SF 布局）、
> 第 32 篇（SM90 FP8）、第 35 篇（SM100 MXFP4/MXFP8）、第 2/3/10 篇（Mega MoE 权重/激活量化）
> 里的每个 M/N/K 语义最终都锚定在这套纯 PyTorch 参考上。
>
> **平台**：这套参考实现全是张量运算，**架构无关，H100 可跑**。本文给出真实实测（`harness/exp_reference_quantizer.py`）。
>
> 源码坐标（`~/github/DeepGEMM`，公开版 commit `057ca59`，版本 2.8.1）：
> - `deep_gemm/utils/math.py:13-16` `ceil_to_ue8m0`
> - `deep_gemm/utils/math.py:19-23` `pack_ue8m0_to_int`
> - `deep_gemm/utils/math.py:26-49` `per_token_cast_to_fp8`
> - `deep_gemm/utils/math.py:52-59` `per_channel_cast_to_fp8`
> - `deep_gemm/utils/math.py:62-72` `per_block_cast_to_fp8`
> - `deep_gemm/utils/math.py:84-96` `_quantize_to_fp4_e2m1`
> - `deep_gemm/utils/math.py:99-125` `per_token_cast_to_fp4`
> - `deep_gemm/utils/math.py:128-141` `transpose_packed_fp4`
> - `deep_gemm/utils/math.py:144-176` `_dequantize_from_fp4_e2m1` / `unpack_ue8m0_from_int` / `cast_back_from_fp8|fp4`
> - 调用形态：`tests/generators.py:317-346`、`tests/test_mega_moe.py:44-56/108`、`tests/test_attention.py:60-68`

## 1. 为什么要有「参考量化器」

DeepGEMM 的每个 kernel 测试都遵循同一个套路:**先用这套参考函数把 bf16/fp32 输入量化成
FP8/FP4 + block scale，再喂给 kernel，最后和「在量化后的操作数上做精确 GEMM」比较**。
也就是说，kernel 的**正确性判据**不是「近似等于原始 bf16 结果」，而是「精确等于量化操作数上的结果」。
这样做的好处是把「量化本身的误差」和「kernel 的实现误差」**解耦**，让测试能抓到真正的实现 bug。

`tests/test_attention.py:60-68` 把这一点写得最直白：它用 `cast_back_from_fp4/fp8` 造一个
`*_simulated` 参考，把 kernel 误差与量化误差分层报告。本文第 §7 的端到端实验就是同一手法。

因此这套函数必须**逐位可复现**、且语义与硬件指令一致——下面逐个拆。

## 2. FP8 家族

### 2.1 `ceil_to_ue8m0`：把 scale 向上取整到 2 的幂

```python
def ceil_to_ue8m0(x):
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)
```

（`math.py:13-16`）它**完全在 fp32 的位域上操作**：

- 取 `|x|` 后拿指数域 `(bits>>23)&0xFF`；
- 只要尾数非 0（`(bits & 0x7FFFFF).bool()`），指数 **+1** ——这就是「进位到下一个 2 的幂」；
  若 `x` 本身是 2 的幂（尾数 0），指数不变，**返回 `x` 自己**；
- `exp` 钳位到 `[1, 254]`：fp32 指数域 0（0 / 次正规）→ 1（即 `2^-126`）；255（inf/nan/溢出）→ 254（`2^127`）。

所以 `ceil_to_ue8m0(x)` = 「**>= |x| 的最小 2 的幂**」，且在正常范围内恒满足
`ceil_to_ue8m0(x) >= |x|` 且结果为 2 的幂。这正是它名字里 **UE8M0**（Unsigned Exponent 8-bit, Mantissa 0）
的含义：scale 只能取 2 的幂，只有 8 位指数。

**为什么要上取整**（而不是就近取整）：因为 SM100 的 SF 走的是 UE8M0——只保留 8 位指数、丢掉尾数与符号
（第 25 篇 `docs/25` §4 讲过，`pack_ue8m0_to_int` 里 `assert (x_int & 0x7fffff == 0)` 会**静默**丢掉非幂次 scale）。
如果先按 `amax/448` 得到非幂次 scale 再取整，scale 只会**变大**（上取整），量化值 `x/sf` 只会**变小**，
永远不会溢出 FP8 的动态范围——这是「安全」的方向。代价是量化误差略增，实测见 §6。

### 2.2 `pack_ue8m0_to_int` / `unpack_ue8m0_from_int`：4 字节打包

```python
def pack_ue8m0_to_int(x):
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    x_int = x.view(torch.int)
    assert (x_int >= 0).all() and (x_int & 0x7fffff == 0).all()
    return (x_int >> 23).to(torch.uint8).view(torch.int)
```

（`math.py:19-23`）把每个 scale 的 8 位指数取出来（`>>23` 后的低 8 位），4 个构成 4 个字节，
`view(torch.int)` 按**小端序**重解释成 1 个 int32：第 0 个 scale 落在**最低**字节（LSB-first）。
反操作 `unpack_ue8m0_from_int`（`math.py:151-152`）是 `(packed.view(uint8) << 23).view(float)`。

这正是第 26 篇讲的 K-grouped SF 打包格式、以及 SM100 GEMM 里 `sf_id` 寻址用的打包单元。
注意两个断言：`x` 必须非负、且尾数为 0（即必须是 2 的幂）——**上游必须先 `ceil_to_ue8m0`**。

### 2.3 `per_token_cast_to_fp8`：1×128 粒度（Mega MoE 激活/权重的默认）

```python
m, n = x.shape
padded_n = align(n, gran_k)
x_padded = torch.empty((m, padded_n), ...).fill_(0); x_padded[:, :n] = x
x_view   = x_padded.view(m, padded_n // gran_k, gran_k)
x_amax   = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
sf       = x_amax / 448.0
sf       = ceil_to_ue8m0(sf) if use_ue8m0 else sf
x_fp8    = (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
```

（`math.py:26-49`）要点：

- **粒度**：每行 `n` 沿最后一维切成 `gran_k`（默认 128）一组，一组一个 scale，输出 `sf` 形状 `(m, ceil(n/gran_k))`。
- **`n` 先补零到 `gran_k` 对齐**（`align`，`math.py:9-10`），但返回时 `[:, :n]` 又把补出来的列切掉。
- **amax 下限 `clamp(1e-4)`** 避免除零；**常数 `448`** 是 E4M3 的最大可表示值（`448 = 1.75×2^8`）。
- `use_ue8m0=True` 时先 `ceil_to_ue8m0`；`use_packed_ue8m0=True` 时再把 scale 按 4 个打包成 int32，
  且**不足 4 的倍数补 1.0**（`math.py:39-48` 的注释说明：补的 scale 覆盖不到任何真实元素、也不会被 kernel 读到）。
- 最终 `.to(torch.float8_e4m3fn)` 是 PyTorch 的 **RNE（round-to-nearest-even）** 转换。

Mega MoE 的激活/权重正是用这条路径量化的：`tests/test_mega_moe.py:54`
`per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)`（注意 **gran_k=32**，
即 MXFP8 的 1×32 粒度），权重同理 `test_mega_moe.py:108` 用 `per_token_cast_to_fp4(..., gran_k=32)`。

### 2.4 `per_channel_cast_to_fp8` 与 `per_block_cast_to_fp8`

- `per_channel_cast_to_fp8`（`math.py:52-59`）：粒度沿**第 0 维**（每组 `gran_k` 行共享一个 scale），
  输出 `sf` 形状 `(m/gran_k, n)`。对应「一列 token 一个 scale」的沿 M 分组。
- `per_block_cast_to_fp8`（`math.py:62-72`）：**128×128 块**粒度。先把 `(m,n)` 两边都补零到 `gran_k` 对齐，
  `view(-1, gran_k, W, gran_k)` 后对 dims `(1,3)` 取 amax，输出 `sf` 形状 `(m/gran_k, n/gran_k)`。
  这是 SM90 1D2D 的 **SFB（权重侧 per-128×128 块）**语义（第 32 篇）。

`per_custom_dims_cast_to_fp8`（`math.py:75-81`）是通用版：给定要保留的维度，对**其余维度**取 amax——
`test_attention.py` 用它给 KV 侧造 scale。

## 3. FP4 家族（E2M1）

### 3.1 `_quantize_to_fp4_e2m1`：手写量化与**平局规则**

```python
ax = x.abs()
# {0, 0.5, 1, 1.5, 2, 3, 4, 6}; midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
code = torch.zeros_like(x, dtype=torch.uint8)
for boundary, round_up in zip((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
                              (False, True, False, True, False, True, False)):
    code += (ax >= boundary if round_up else ax > boundary).to(torch.uint8)
sign = (x < 0) & (code != 0)
code = code | (sign.to(torch.uint8) << 3)
return code.view(torch.int8)
```

（`math.py:84-96`）E2M1 的 8 个正数值是 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`（编码 0..7），
相邻值的**中点**就是那 7 个 `boundary`。`code` 记录「越过了几个中点」，就得到最近编码：

- **平局**（`ax` 恰等于某个中点）时，用 `round_up` 标志决定进位与否。实测这 7 个标志让平局
  「取编码为**偶数**的那个」：`0.25→0(down)`、`0.75→2(up)`、`1.25→2(down)`、`1.75→4(up)`、
  `2.5→4(down)`、`3.5→6(up)`、`5.0→6(down)`——即 **ties-to-even（按 E2M1 编码）**，与硬件
  `cvt.rn` 的取整语义一致。§6 实测坐实。
- **符号位**：`sign = (x<0) & (code!=0)`——**不用「负零」**（`-0.0` 编码成 0 而非 8），避免多一个冗余编码。

**为什么手写而不直接用 `torch.to(float4)`**：PyTorch 对 E2M1 的支持不完整/不统一，参考实现必须自己
精确控制编码与平局，才能作为 kernel 的判定基准。

### 3.2 `per_token_cast_to_fp4` 与 nibble 打包

- 结构同 `per_token_cast_to_fp8`，但 **`sf = amax/6`**（`6` 是 E2M1 最大可表示值，`math.py:109`），
  且 `gran_k` 默认 128（Mega MoE 用 32）。
- `_quantize_to_fp4_e2m1` 得到每个元素的 int8 编码后，**两个 nibble 打成一个字节**：
  `packed = codes[:, even] & 0x0F | (codes[:, odd] & 0x0F) << 4`（`math.py:113-114`）——
  **偶数下标在低 4 位（LSB-first）**。输出形状 `(m, n/2)`。

### 3.3 `transpose_packed_fp4`

（`math.py:128-141`）先解包成 `(m,n)` 的编码，`transpose(0,1).contiguous()` 后按新行两两重打包成
`(n, m/2)`。用于把 K-major 的 FP4 权重转成 MN-major（`tests/generators.py:320`）。要求 `m` 为偶数。

## 4. 反量化：`cast_back_from_fp8` / `cast_back_from_fp4`

- `cast_back_from_fp8`（`math.py:170-176`）：`x_fp8.float() * sf[:, group_idx]`，其中
  `group_idx = arange(n)//gran_k`——即 per-token 的按列分组重建。
- `cast_back_from_fp4`（`math.py:155-167`）：解包 nibble（也是 LSB-first：`unpacked[:, ::2]=lo, [:,1::2]=hi`），
  `_dequantize_from_fp4_e2m1` 查表 `{0,0.5,...,6}` 并按符号位取负，再乘 scale。
- `use_packed_ue8m0=True` 时先用 `unpack_ue8m0_from_int` 把 int32 还原成 scale。

这两个函数是测试的「模拟参考」来源（`test_attention.py:66-68`）：**它们把量化误差从 kernel 误差里隔离出来**。

## 5. 在 Mega MoE 测试里怎么被使用

`tests/test_mega_moe.py` 是端到端 Mega MoE 测试（**SM100 专属，本机不跑**，第 20/2 篇）。
它对每个 rank 的隐藏状态与专家权重调用参考量化器，再喂给 `fp8_fp4_mega_moe`：

- 激活：`per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)`（`:54`），
  再把打包好的 SF **重排成 kernel 要的交错行序**（`:40-56` 的
  `transposed_m_idx = (m_idx//128)*128 + (m_idx%32)*4 + (m_idx%128)//32`——就是第 2 篇讲的 UTCCP 4x32 转置）。
- 路由权重：`per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)`（`:108`）
  或 FP8 版（`:117`），再 `transform_sf_into_required_layout(..., (1,32), num_groups)`（第 26 篇）。
- K-grouped wgrad 的输入也由参考量化器逐组构造（`tests/generators.py:500-525`）。

所以「Mega MoE kernel 对不对」这件事，在这套测试里等价于「kernel 是否精确复现了参考量化器定义的操作数上的 GEMM」。

## 6. H100 实测

`harness/exp_reference_quantizer.sh`（→ `exp_reference_quantizer.out.txt`），H100 80GB / cc 9.0，
纯张量运算，**ALL PASS**：

### A. `ceil_to_ue8m0`（`math.py:13-16`）

- 1040 个样本（`2^-140..2^130` 的幂次 + 幂次附近抖动 + 0/次正规/溢出/inf/nan）与**独立参考**
  （用 `math.frexp` 算「>= 的最小 2 的幂」再钳位 exp 域）**exp 域 0 不符**；输出**恒为 2 的幂**。
- 边界（实测）：`ceil_to_ue8m0(0)=2^-126`、`ceil_to_ue8m0(2^-125)=2^-125`（幂次原样）、
  `ceil_to_ue8m0(3·2^-128)=2^-126`、`ceil_to_ue8m0(1.5)=2`、`ceil_to_ue8m0(2)=2`、
  `ceil_to_ue8m0(1.7e38)=1.7e38`、`ceil_to_ue8m0(1.87e38)=1.7e38`（被 254 钳位）、
  `ceil_to_ue8m0(inf)=1.7e38`。
- **唯一「输出 < |x|」的 8 个样本都在溢出区**（`|x| > 2^127`，被钳到 `2^127`）——正常范围内恒 `>= |x|`。

### B. `pack_ue8m0_to_int` / `unpack`（`math.py:19-23/151-152`）

- 指数 `[124, 132, 138, 132]` → int32 `0x848a847c`，与手算 LSB-first（`0x7c | 0x84<<8 | 0x8a<<16 | 0x84<<24`）**一致**。
- 往返逐位一致；4096×128 随机 scale 打包→解包**逐位一致**。

### C. FP8 三种粒度（`math.py:26-72`）

同一份数据（`256×1024`，每 4 行放大约 8× 制造尺度差异）：

| 粒度 | `use_ue8m0=False`（max abs err / max\|x\|） | `use_ue8m0=True` | sf 形状 |
|---|---|---|---|
| per_token 1×128 | 2.9774e-02 | 4.1483e-02 | (256, 8) |
| per_channel | 2.8112e-02 | 4.1483e-02 | (2, 1024) |
| per_block 128×128 | 3.4662e-02 | 4.1483e-02 | (2, 8) |

- 全零块 SF = `2.232e-07` = `clamp(amax,1e-4)/448`（常数 448 与 1e-4 地板都被坐实）。
- 把 scale **上取整到 UE8M0** 后 relerr `2.1955e-02 → 2.2613e-02`（略增，换来 scale 只有 8 位指数、可打包）。

### D. FP4 E2M1 取整/平局（`math.py:84-96`）

7 个中点取整方向实测：

```
0.2500 -> 0 (0.00) down   1.2500 -> 2 (1.00) down   2.5000 -> 4 (2.00) down   5.0000 -> 6 (4.00) down
0.7500 -> 2 (1.00) up     1.7500 -> 4 (2.00) up     3.5000 -> 6 (4.00) up
```

- 7 个中点**全部等于 ties-to-even（按 E2M1 编码）**；**20 万随机样本 vs 独立参考（最近值 + 偶数平局）0 不符**。
- 对照 ties-away（平局一律向上）只在 7 个中点里的 **4 个**不同——坐实平局规则就是 ties-to-even。

### E. FP4 nibble 打包与往返（`math.py:99-125`）

- `codes [1,2,3,4,5,6,7,9,15,0]` → 字节 `0x21,0x43,0x65,0x97,0x0f`：第 0 个编码在**低 nibble**（LSB-first）。
- FP4 往返（含 UE8M0 SF）relerr `1.2384e-01` vs 同数据 per-token FP8 `2.2504e-02`（约 **5.5×** 粗，
  E2M1 只有 3 位有效 vs E4M3）。

### F. `transpose_packed_fp4`（`math.py:128-141`）

- `(64,48) → (96,32)`，与「解包→转置→重打包」的独立参考**逐位一致**；且**对合**（转置两次回到原值）。

### G. 端到端：参考量化器定义 kernel 输入

用参考量化器量化 `a`（per-token FP8）/`b`（per-block FP8）后真跑 SM90 `fp8_gemm_nt`（`m=n=k=1024/512/1024`）：

- kernel vs 「参考量化器**反量化后 fp64 精确 GEMM**」relerr **1.4217e-03**（只剩 bf16 输出舍入）
  → 坐实 kernel 的输入语义**就是**这套参考；
- 同一 kernel vs 「原始 bf16 精确 GEMM」relerr **3.6969e-02**（多出来的是 FP8 per-token/per-block 量化噪声）。

## 7. 结论与边界

- `deep_gemm/utils/math.py` 是 DeepGEMM 全部 FP8/FP4 kernel 的**量化语义基准**：
  `ceil_to_ue8m0` 上取整到 2 的幂（UE8M0）、`pack_ue8m0_to_int` 4 字节 LSB-first、
  `per_token/block/channel` 三种粒度的 `448`（E4M3）/`6`（E2M1）常数与 1e-4 地板、
  `_quantize_to_fp4_e2m1` 的 **ties-to-even（按编码）**、nibble 的 LSB-first 打包、`transpose_packed_fp4` 的对合性。
- **都不是 SM100 专属**：纯张量运算，H100 上 A–G 全部实测通过；这套基准是理解第 25/26/32/35 篇
  里所有 SF 布局与量化分支的「地基」。
- 边界：`ceil_to_ue8m0` 在 `|x| > 2^127` 时会**低估**（钳到 `2^127`），但真实 amax/448 远小于该阈值；
  `cast_back_from_fp8` 只实现 per-token 行分组，per-block/per-channel 需自行按 128 展开（本文实验脚本即如此）。
