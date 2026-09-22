# 用 SM90 的 FA3 重新对比：FA3 vs FA2 vs TE（更正上一篇的结论）

> `docs/06` 用 `flash_attn` **2.7.4** 对标 TE，得出「FA 比 TE 慢 1.2–1.6×」，并归因到
> 「FA2 = SM80 内核」。**这其实是不公平的**：H100 上应当用 **FA3（`flash_attn_3`，SM90 wgmma+TMA）**。
> 本文把 FA3 编译出来重新测——**结论反转为：FA3 比 TE 更快**。
> 原始输出 `src/fa2_fa3_te_bwd_only.out.txt`；探针 `harness/fa3_probe.py`。

---

## 1. 编译 FA3（`flash_attn_3`, SM90）

- 源码：`~/github/flash-attention/hopper/`（包名 `flash_attn_3`，`flash_attn_3.flash_attn_interface`）。
- 依赖 CUTLASS：仓库把 `csrc/cutlass` 作为 submodule，但本机 submodule 拉取失败（网络），
  故用本地 `~/github/cutlass`（4.8.0）**符号链接**到 `csrc/cutlass`，并 `git config submodule.csrc/cutlass.update none`。
- 只编译需要的实例（否则 451 个 `.cu` 太久）：
  ```
  FLASH_ATTENTION_FORCE_BUILD=TRUE
  FLASH_ATTENTION_DISABLE_SM80=TRUE
  FLASH_ATTENTION_DISABLE_HDIM64/96/192/256=TRUE
  FLASH_ATTENTION_DISABLE_HDIMDIFF64/192=TRUE
  FLASH_ATTENTION_DISABLE_LOCAL/SOFTCAP/PAGEDKV/APPENDKV/FP8=TRUE
  MAX_JOBS=64 python setup.py install
  ```
  实际只编译 **11 个目标**，几十秒完成。产物 `flash_attn_3 3.0.0`（`import flash_attn_3` 成功）。
- 数值校验（`harness/fa3_probe.py`，vs fp32 ref，fp16）：MHA `(1,512,16,128)` o/dq/dk/dv maxdiff
  `1.55/1.53/1.67/1.86e-3`；GQA `(1,1024,40,128) kv8` `1.49/1.67/3.30/3.80e-3`；
  MQA `(1,1024,64,128) kv1` `1.58/2.35/6.35/7.64e-3` —— 与 FA2/TE 同量级，**FA3 可用且正确**。

## 2. 纯反向性能：FA2 vs FA3 vs TE（CUPTI，bf16 与 fp16 一致）

| 形状 (B,S,H,D) | FA2.7.4 (SM80) | **FA3 (SM90)** | TE2.14 (cuDNN SM90) | FA3/FA2 |
|---|---|---|---|---|
| (1,1024,40,128) kv=8 GQA | 230 TF | **355 TF** | 325 TF | 1.54× |
| (1,1024,32,128) kv=4 GQA | 217 TF | **418 TF** | 307 TF | 1.93× |
| (1,1024,64,128) kv=4 GQA | 258 TF | **431 TF** | 355 TF | 1.67× |
| (1,1024,64,128) kv=1 MQA | 258 TF | **438 TF** | 321 TF | 1.70× |
| (1,4096,16,128) MHA | 377 TF | **850 TF** | 618 TF | 2.25× |
| (1,256,2,512) MLA | 不支持 | 不支持 | 不支持 | — |
| (1,512,4,512) MLA | 不支持 | 不支持 | 不支持 | — |
| (1,1024,2,512) MLA | 不支持 | 不支持 | 不支持 | — |

（单位 ms/TFLOPS 原表见 `src/fa2_fa3_te_bwd_only.out.txt`。）

**更正后的结论**：
- **FA3 在 H100 上比 TE 更快**：GQA/MQA 快 ~8–36%，大 MHA 快 ~38%（850 vs 618 TF）。
- FA3 相对 FA2 快 **1.5–2.25×**——这正是 SM90 `wgmma`+TMA 相对 SM80 `mma.sync`+`cp.async` 的代际差。
- **MLA（head_dim=512）三者都不支持反向**：FA2 限 ≤256，FA3 同（报 `mha_fwd` 错），TE 训练 bwd 限 ≤256。
  该形状的反向数字**只能由 ours 提供**（已支持 fp16）。

## 3. ncu 证据：FA3 与 TE 是同一代指令（wgmma + TMA）

**kernel 级**（(1,4096,16,128)，一次反向）：

| | 反向主 kernel | 时间 |
|---|---|---|
| FA2 | `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<...>>` | 683 µs |
| FA3 | `flash::enable_sm90<flash::FlashAttnBwdSm90<flash::CollectiveMainloopBwdS...>>` | **283 µs** |
| TE | `cudnn_generated_..._sm90_flash_bprop_`**`wgmma`**`_f16_...` | 397 µs |

**SASS 指令统计**（主反向 kernel）：

| 指令 | FA2 | **FA3** | TE |
|---|---|---|---|
| `HMMA.16816.F32`（Ampere mma.sync） | 320 | 0 | 0 |
| `LDSM`（ldmatrix） | 144 | 0 | 0 |
| `LDGSTS`（cp.async） | 32 | 0 | 0 |
| `UTMALDG` / `UTMASTG` / `UBLKCP`（**TMA**） | 0 | **20 / 4 / 8** | 8 / 4 / — |
| `WARPGROUP.ARRIVE/DEPBAR`（**wgmma** 同步） | 0 | **5 / 5** | 5 / 5 |
| `STSM`（stmatrix） | 0 | 20 | 20 |

**SOL 对照**（主反向 kernel，S=4096）：

| 指标 | FA2 | **FA3** | TE |
|---|---|---|---|
| Duration | 682 µs | **289 µs** | 399 µs |
| Compute (SM) | 39.8% | **67.6%** | 46.7% |
| DRAM Throughput | 6.6% | 15.4% | 11.3% |
| L1/TEX Throughput | 72.2% | 75.6% | 63.2% |
| Achieved Occupancy | 12.5% | 15.3% | 15.6% |

FA3 与 TE 都是 Hopper 内核：**TMA 搬数 + warpgroup(wgmma) 计算 + stmatrix 回写**，
`Compute` 利用率 67.6% 明显高于 FA2 的 39.8%、TE 的 46.7%——这就是它最快的原因。

## 4. 方法论：对标要选对「同代实现」

- **答用户的问题**：「FA 比 TE 慢」的根因是**选错了 FA 的版本**：pip 装的 `flash_attn` 2.7.4 是 **FA2（SM80）**，
  在 H100 上本就不该拿它对标 TE 的 SM90 内核。**正解是 FA3**（源码在 `hopper/`）。
- FA3 需要**从源码编译**（pip 没有通用 wheel），依赖 CUTLASS（可符号链接本地 clone），
  用 `FLASH_ATTENTION_DISABLE_*` 裁剪实例可把编译从 451 个文件降到 11 个。
- **口径**：统一用纯反向（`harness/fa_vs_te_bwd_only.py`），CUPTI 纯 device 时间，同 shape、同 dtype、同 causal。
- **MLA head_dim=512**：三套参考实现（FA2/FA3/TE）都不支持反向，**ours 是唯一能给出数字的**。

## 5. 对 ours 的启示（更正 docs/06 §5）

1. ours 的补课目标从「追上 FA2」升级为「**对标 FA3/TE 的 Hopper 路线**」：
   fp16/bf16 先 `mma.m16n8k16`+`ldmatrix`（O5，10–20×），再 **`wgmma`+TMA+warp specialization**（O9）；
2. **GQA/MQA 的 KV 归约要放进 kernel**（FA2 单开 `reduce` 吃亏；FA3/TE 在 kernel 内做）；
3. 基准与结论一律用**纯反向 + 同代对标**，并在文档里写明对标的是 FA2 还是 FA3。
