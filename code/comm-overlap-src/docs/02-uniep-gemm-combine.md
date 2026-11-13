# （二）UniEP 的 FC2 + Combine：算完一个 tile 就发出去

**combine 把「GEMM 算完再统一发送」换成了「一个 tile 算完就按 token 归位推走」，代价是 push 慢会反压下一轮 GEMM**。

读完你会知道：

- 为什么没有「通信 warp」，push 是谁顺手做的。
- 一行结果凭什么知道去哪个 rank 的哪个槽。
- 哪个同步点省不掉。

dispatch 那半（[01](01-uniep-dispatch-gemm.md)）等一个 expert：行没齐就算不了。这半只等一个 tile：一行只属于一个目标 rank，写过去就完事，不用等齐。

## 1. 少掉两个 warp，不是省资源

| | MegaKernel ①（dispatch + FC1） | MegaKernel ②（FC2 + combine） |
|---|---|---|
| warp 数 | 8（epilogue 0-3 / dispatch 4-5 / MMA 6 / TMA 7） | 6（epilogue 0-3 / MMA 4 / TMA 5） |
| 通信方向 | pull（TMA 从对端读） | push（SIMT 128-bit store 写到对端） |
| 同步点 | 每个 expert 一个 signal | 无 per-expert 同步，只有 kernel 后一次 group barrier |

① 用 2 个 warp 做 pull，② 少掉的正是它们，不是省资源：push 塞进已持有累加器的 epilogue lane，本来就要搬 TMEM 结果，顺手写去对端，不需要第三个角色。SMEM 只花在 GEMM stages。

对端 buffer 可直接写：gpu 域的 128b store，`l1c_evict_priority` 设 `NO_ALLOCATE` 绕开 L1（只写一次，绕 L1 省带宽不省延迟），不用落 SMEM 再 S2G。

## 2. 每行结果自己带着回程地址

看什么：把行号翻成槽位。

```python
my_global_row = global_m_base + epi_warp_idx * cpy_m_per_warp + epi_lane
encoded = token_src_rank_topk_and_indices[my_global_row]
src_token_idx, my_dst_rank, src_topk_idx = decode_token_src_rank_topk_and_indices(encoded)
my_dst_slot = src_token_idx * topk + src_topk_idx
```

所以呢：A 的第 `my_global_row` 行是源 rank 的 token `src_token_idx` 的第 `src_topk_idx` 个专家槽，落到对端 staging 的 `my_dst_slot` 行。槽位只跟 token 有关，不查路由表，只认 dispatch 随行带的编码。

- `lane ↔ row` 一一对应（`cpy_m_per_warp == 32`、`epi_tile_m == cta_tile_m`），epilogue 不用再 cross-lane 归约。
- 16 个 lane 合作搬一行（每 lane 128 bit = 8 个 bf16），一行的 N 方向切成 `N/8` 个 atom。
- 丢弃槽（`dst_rank == -1`）先兜底再折地址：`safe_dst_rank` 落到 rank 0 / slot 0，`my_pred` 记 0/1，地址乘上它算成 0。0 页不映射，故这个假设只在统一寻址下成立。

写出去只有一步：leader 用 `shuffle_sync` 广播行基址给 16 个 lane，`bk_row_base != 0` 挡掉地址为 0 的行。

## 3. 单槽 SMEM 换一个 A/B stage

看什么：常规 epilogue 开 `num_epi_stage = 2+` 槽做 R2S/TMA 交替；这里强制 `num_epi_stage == 1`，R2S 与 push 按 subtile 交错：

```text
for each work tile:
  for s in subtile_cnt:
      TMEM → RMEM (cast bf16)
      if s == subtile_cnt - 1: release acc stage    ← 关键
      RMEM → SMEM[0]
      SMEM[0] → peer GMEM (cooperative push)
```

- WAR 冲突（下一轮 R2S 覆盖 SMEM[0] 时 push 读完没）靠 per-warp program order + 一次 `sync_warp`：写者读者是同一 warp 的同一批 lane；拆多槽或跨 warp 要补 barrier。
- 所以呢：省下的 SMEM = `(subtile_cnt - 1) × epi_bytes_per_stage`，正好够在 M=128/N=256/K=2560 的 bf16 主负载上多开一个 A/B stage——用调度复杂度换流水深度。

## 4. 重叠的是下一个 tile 的 MMA

看什么：【时序】算完一个 tile → 推出去 → 下一个 tile 开算，中间没有全量 barrier，关键只有一句：

```python
if subtile_idx == subtile_cnt_const - 1:
    with elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)   # 在 push 之前
```

所以呢：累加器在最后一个 subtile 读完 TMEM 就还给 MMA，不等 push 结束——MMA 就能在 epilogue 还在推时开下一个 tile。本 kernel 唯一的计算-通信重叠是 push 与下一个 tile 的 MMA，不是 dispatch 与当前 tile。

- 【代价】领先最多一个 tile：acc stage 只有 2 个（`num_acc_stage=2`），push 太慢会反堵 MMA。
- 这个 2 是 TMEM 定的：`(128,256)` 的 fp32 累加器一级占满 256 列，两级吃满 512 列——深度卡在 TMEM，不是 SMEM。

权重搭同一趟车：dispatch 把每个 token-slot 的权重搬到本地，combine 时按 `dst_slot` 推去对端，`n_tile_idx == 0` 保证多 N-tile 只写一次；它放在 `consumer_wait` 前，和等 acc 的停顿共处，等于白捡。

## 5. 唯一省不掉的是最后一次 barrier

`topk_reduce` 要读回 staging 做 top-k 求和，得知道所有 rank 都推完，靠 host 侧 `group_gemm_combine` 的一次 `ep_group_barrier()` 收口。

> UniEP 的 overlap 边界止于 combine 的 push。归约是一段串行区，它前面的 barrier 长度 ∝ 最慢 rank 的 combine 时间。

这段串行区里每个 token 的 k 份贡献必须按固定顺序求和，否则每次跑出来的低位都不一样——确定性单独讲，就为这个。

## 6. 【边界】细粒度什么时候反噬

- 发射开销：push 是 SIMT store，逐条发射，没有 TMA 描述符搬块的压缩，一条 store 只搬 8 个 bf16。tile 取小就摊不薄，push 一慢就反压 MMA。
- 跨节点：对端 buffer 是当本地地址写的（`output_ptrs` 给基址，gpu 域的 store），没有「写不过去」的第二条路径。这条路不成立，上面的时序作废。
- 断言：`num_epi_stage == 1` 是 assert 出来的（信息写着 `requires num_epi_stage==1`），改 stage 数不是回退，是不合法。

## 7. 小结

如果只记一件事：combine 快不在通信，而在等待的颗粒度更小——只等一个 tile。

上限不在 SMEM，在 TMEM 的 512 列：acc 只有 2 档。

下一篇看这段串行区怎么做到可复现：[03](03-uniep-determinism.md)。

对应副本：[`cutedsl_group_gemm_combine.py`](../uniep/kernels/cutedsl_group_gemm_combine.py)（上游 1067 行 → 副本 1411 行，972 行代码；行号易漂，故不引。）
