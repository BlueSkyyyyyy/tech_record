# 94 · DeepEP bucket 混合集合算子的验收测试即正确性规范

> 专题的 DeepEP 侧已经把 bucket 系列拆得很细：第 64 篇（P54-1）拆了三类算子 × 三个物理域的
> `BucketBuffer`、copy-engine 与分块；第 62 篇（P62-1）拆了 `all_reduce`/`reduce_scatter` 的三后端
> 与信用环、bf16 随机舍入；第 83 篇（P83-1）拆了 `BucketSession` 的会话机制与生命周期守卫。
> 但把它们组织成「**一套可执行的正确性规范**」的两份旗舰测试——`tests/bucket/test_mixed.py`
> 与 `tests/bucket/test_perf.py`——一直只被顺带引用几行。本篇把它们当 bucket 家族
> 「算什么对 / 怎么比」的**规范**来拆：mixed 定义正确性（op/精度/物理域门控、bucket 尺寸量化、
> 通信组拓扑、输入与 guard 构造、NCCL 参考、逐位/容差判据、流交错与销毁），perf 定义基准
> （NCCL profile 矩阵、coalescing、ragged bucket、中位数计时）。

本篇目标源码（`~/github/DeepEP@93eb6eb`）：

- `tests/bucket/test_mixed.py`（373 行）：`make_groups:44-66`、`supported:69-79`、
  `make_sizes:82-88`、`make_schedule:91-102`、`make_inputs:105-123`、`nccl_reference:126-141`、
  `verify:144-154`、`run_round:157-314`、`test:317-344`、`main:347-373`。
- `tests/bucket/test_perf.py`（273 行）：常量 `:28-48`、`measure:51-73`、`nccl_collective:76-84`、
  `get_nccl_sm_budget:87-88`、`make_sizes:91-98`、`get_skip_reason:101-106`、
  `run_profiles:218-249`。
- 消费方：`deep_ep/buffers/bucket.py` 的 `get_theoretical_num_sms:236`、`session:299`、
  `barrier:304`、`reduce_scatter:313`、`all_reduce:334`、`all_gather:361`、`destroy:394`；
  `deep_ep/comm/handle.py:80` 的 `get_physical_domain_size`；`csrc/buffers/bucket.hpp:151/219/284`
  的注册断言。

**可跑边界**：`run_round`/`test` 要构造 `deep_ep.BucketBuffer`，依赖 DeepEP 的 C++/NCCL 设备 API
（`ncclDevComm`/`ncclGin`）——本机 NCCL 2.29.7 不匹配（第 86 篇「阻塞」），`import deep_ep` 即 abort，
故 **buffer 端到端仅分析**。但测试侧的规范逻辑是**纯 Python / 纯张量**：`supported`、`make_sizes`、
`make_schedule`、`make_groups`、`make_inputs`、`nccl_reference`、`verify`、`get_skip_reason`、
`measure`。这些在 H100(sm90) 上原样抽取运行、与独立 oracle 对拍，并用 8×H100 真机 NCCL 坐实
`nccl_reference` 语义与 `rs_ag` 不变量（`harness/exp_deepep_bucket_test_oracle.{sh,py,out.txt}`，
**33 PASS / 0 FAIL**）。

---

## 1. 目标：mixed 测的是「混合」，不是「单个算子」

单个算子的语义早在第 54/62 篇讲过（`all_gather` 搬运、`all_reduce`/`reduce_scatter` 归约）。
`test_mixed.py` 的独特之处是把它们**混在同一轮里随机编排**，专门暴露交互错误。它在模块顶部就
声明了这个意图——`OPS = ('all_reduce', 'reduce_scatter', 'all_gather', 'rs_ag')`（`:24`），其中
`rs_ag` 不是第四种硬件算子，而是**「先 reduce_scatter 再 all_gather」的组合**（`run_round:244-265`
两个 `buffer.*` 调用拼出来），用来验证「两个半程组合 == 一个整程」。

`run_round`（`:157-314`）在多轮里轮换一堆正交维度：

| 维度 | 位置 | 取值 / 语义 |
|---|---|---|
| 销毁模式 | `:164` | `round_idx % 2 == 0` → `explicitly_destroy=True` |
| 会话 | `:165` | 每 3 轮用一次 `BucketSession`（临时 scratch） |
| 计算流 | `:167-169` | default / 普通 / 高优先级 / 普通，构造期随机选一个并 `wait_stream` |
| 用例 | `:189-193` | schedule 随机洗牌，用尽后从 schedule 里重复随机取 |
| 精确判据 | `:196` | bf16 / `all_gather` / 每 7 步 → `exact=True` |
| 缩放 | `:198` | `{1.0, 0.0, 0.25, -0.5, 2.0}`（`all_gather` 恒 1.0） |
| 计算 SM | `:200` | `all_gather` 必须 0（copy engine 驱动、无 SM kernel） |
| bucket 数 | `:202` | 首步全用、之后 `1 + step % num_buckets` |
| packed | `:205` | 单 bucket 时按周期把 list 折成单个张量传 |
| 三条流 | `:207` | producer / launch / consumer **各自独立**随机 |
| dtype | `:209` | `all_gather` 试 `{fp32,bf16,int32,uint8}`，归约恒 fp32 |
| 外部输入 | `:214` | 纯 NVLink `all_gather` 且非会话时，每 4 步用纯外部 src |

「三条流各自独立」是这一测试最狠的一维：producer 写输入、launch 发集合、consumer 取结果必须在
**互不相同的流**上仍保持依赖（`:228-236` 用 event 串起 previous-consume → input-write → launch）。
这正是第 65 篇（P65-1）「通信-计算重叠的调度地基」在**测试侧**的对应物。

---

## 2. 能力门控：`supported` 与 `get_skip_reason`

同一份 schedule 必须在所有 rank 上一致，所以「哪些组合能跑」必须是一个**确定性的纯函数**。

`supported(op, physical, precision)`（`:69-79`）以**物理域** `(num_rdma_ranks, num_nvl_ranks)`
为键：

```python
if op == 'all_gather':            return True
if rdma * nvl == 1:               return False      # 单 rank 归约未实现
if op == 'all_reduce':            return True
if precision == 'bf16':           return rdma > 1 and nvl == 1   # bf16 只在纯 RDMA
return True
```

- 单 rank（`rdma*nvl==1`）时**只有 all_gather 能跑**——搬运无需归约；`all_reduce`/`reduce_scatter`
  在这种拓扑下没有 kernel 实现。
- **bf16 通信只在纯 RDMA**（`rdma>1 and nvl==1`）：因为 bf16 归约走「RDMA 反向环 + 随机舍入」
  这条第 62 篇讲过的路径；纯 NVLink 的 `multimem` 归约只支持 fp32。

`test_perf.py` 的 `get_skip_reason`（`:101-106`）是同一套约束的**性能版**（返回原因串而不是布尔）：
单 rank 归约、以及 `reduce_scatter` 的 bf16 非纯 RDMA。八卡同节点 H100 的物理域是 `(1,8)`
（第 63 篇：`get_physical_domain_size` = `rank数/LSA × ncclTeamLsa.nRanks`），于是
`reduce_scatter/bf16` 会被跳过——这正是 `supported` 里 `rdma>1` 那一支在真机上的落点。

---

## 3. bucket 尺寸量化：`make_sizes` 的 `quantum`

`make_sizes`（`:82-88`）是「把用户给的 `num_bytes` 摊成若干个 bucket 尺寸」。它的第一行定义了
**量化步长**：

```python
quantum = math.lcm(*(group.size() * 32 for group in groups))
sizes = [align(min(x * groups[0].size(), num_bytes), quantum) for x in (32, 4064, 32736)]
sizes += [align(rng.randrange(1, num_bytes + 1), quantum) for _ in range(3)]
sizes += [align(num_bytes, quantum)]
sizes += [align(min((i + 1) * quantum, num_bytes), quantum) for i in range(...)]
```

- `group.size() * 32`：每 rank 分到的分片要 32 字节对齐（`float32.itemsize`，第 64 篇
  `SHARD_ALIGNMENT=32`），所以每个 bucket 尺寸必须是 `group_size × 32` 的倍数。
- 多个通信组时取 **lcm**：同一批 bucket 会喂给不同大小的组（world / half / nvl / rail），
  尺寸必须同时整除所有组的分片。
- 三个「故意奇怪」的尺寸 `32, 4064, 32736` 乘上 `groups[0].size()`：`4064 = 32·127`、
  `32736 = 4064·8` 等，都是「比整块少一点 / 卡在分片边界」的尺寸，配 `align` 逼出边界。
- 之后再来 3 个随机尺寸、1 个满尺寸、以及不够就补到 `num_buckets` 个的阶梯尺寸。

`test_perf.py:91-98` 的 `make_sizes` 是性能版：默认等分，`ragged=True` 时**一个满 bucket + 一堆
小 bucket**（`alignment * (i % 7 + 1)`），再 shuffle——专门测桶大小不齐时的负载均衡。

---

## 4. 用例枚举：`make_schedule`

`make_schedule`（`:91-102`）把「组 × op × 精度」展开成用例表，用 `supported` 过滤：`all_gather`/
`all_reduce` 只看 fp32，`reduce_scatter`/`rs_ag` 才试 `--comm-precisions`（fp32/bf16）：

```python
for group_idx, physical in enumerate(physicals):
    for op in args.ops:
        precisions = args.comm_precisions if op in ('reduce_scatter', 'rs_ag') else ['fp32']
        for precision in precisions:
            if supported(op, physical, precision): schedule.append((group_idx, op, precision))
            elif log_skips: print(f'> SKIP: ... {op}/{precision} is not implemented')
assert schedule, 'No supported operations for the selected groups'
```

注意 `all_reduce` 也被排除在 bf16 之外（恒 `['fp32']`）——原因同 §2：NVLink 归约是 fp32 的。
这个 schedule 在 `run_round:189-190` 被 `rng.shuffle` 一次，且 RNG 是 **rank 无关的**
（`:159` 的 `random.Random(seed + round_idx * ROUND_SEED_STRIDE)`，注释明说「Must be used in the
same way by all ranks」），所以所有 rank 拿到同一顺序。

---

## 5. 通信组拓扑：`make_groups`

`make_groups(world, nvl)`（`:44-66`）为每个 rank 造出一组「它属于的」通信组，覆盖三类物理域：

1. **world**（`:52`）：全体。
2. **等分连续段**（`:53-58`）：若 world 是偶数且 `half` 与 `nvl` 互相整除，加 half 段；
   再加 nvl 段（若 `1 < nvl < world` 且不等于 half）。
3. **纯 RDMA rail 组**（`:60-65`）：`nvl` 个「跨节点同序号」的组
   （`range(rail, world.size(), nvl)`）——每组的 RDMA 域都 >1。

`nvl` 就是同节点 NVLink 域大小。八卡同节点时 `(rdma,nvl)=(1,8)`：全都会是「纯 NVLink 大组」；
而跨节点部署（如 2×4）才会生出 `rdma>1` 的 rail 组——这解释了 §2 里 `rdma>1 and nvl==1` 的 bf16
约束**只在多节点时才可能满足**。

---

## 6. 输入与 guard：`make_inputs` 与 `verify`

为了防止 kernel **越界写**，每个 arena 被切成「guard 前缀 + payload + guard 后缀」
（`GUARD_PREFIX_ELEMS=8`、`SUFFIX=24`、`GUARD_BYTES=128`，`:27-29`），整块先填 `GUARD_VALUE=12345`。
`make_inputs`（`:105-123`）从 arena 中切出 payload，并按情形处理：

- **会话模式**：payload 必须是新分配的 `torch.empty_like`（`:110-111`）——因为张量必须注册进 buffer，
  不能直接用 arena 里的视图（第 83 篇）。
- **all_gather**：payload 再切成 `group.size()` 份、取本 rank 那份（`:113-114`）——all-gather 的输入
  是**本地分片**。
- **exact**：用 `randint(-2,3)` 的小整数模板（`uint8` 用 `0..3`），保证浮点/整型都能**逐位**比
  （`:115-116`）。
- 非会话时把两个 guard 视图返回（`:121-122`）。

`verify`（`:144-154`）是判据本身：

```python
done.synchronize()
for actual, expected in zip(snapshots, references, strict=True):
    assert_close(actual.view(-1), expected.view(-1), rtol=0 if exact else 2e-4, atol=...)
for guard in guards:
    assert_close(guard, full_like(guard, GUARD_VALUE), rtol=0, atol=0)
```

两个要点：① `exact=True` 时 `rtol=atol=0`（**逐位**）；② guard 永远逐位等于 `GUARD_VALUE`——
这是「kernel 只动 payload、不碰边界」的可执行保证。`verify` 里任何一个 assert 失败都被包成
`AssertionError(f'Failed case {case}: ...')`，把出错的用例描述带出来。

---

## 7. NCCL 参考：`nccl_reference` 与 `rs_ag` 不变量

`nccl_reference(op, inputs, group, scale)`（`:126-141`）用**系统 NCCL 本身**当参考，对每个输入
副本：

- `all_reduce` / `rs_ag`：`dist.all_reduce` 后乘 `scale`（`:129-132`）。
- `reduce_scatter`：`dist.reduce_scatter_single` 后乘 `scale`（`:133-136`）。
- `all_gather`：`dist.all_gather_single`（`:137-139`，无 scale）。

**为什么 `rs_ag` 的参考走 `all_reduce`**：`rs_ag` 把整段张量先 reduce_scatter 成每 rank 一片、
再 all_gather 拼回整段——对**同一段**张量做这两步恰好等于一次 all_reduce（每个元素被全网求和一次、
回到原位）。这是 kernel 实现 `rs_ag` 组合语义的正确性锚点，也是第 62 篇 `reduce_scatter`+
`all_gather` 组合的理论依据。

**本机差异**：DeepEP@93eb6eb 用的 `dist.reduce_scatter_single`/`all_gather_single` 在本机
torch 2.12.0a0+nv26.04 **不存在**（本机只有 `*_tensor`/`_all_gather_base`），故实验里装了兼容别名
（二者签名一致：`(output, input, group=...)`），原样驱动真实函数体。这本身是一条环境发现：
这份测试要求一个更新（改名后）的 torch。

---

## 8. `run_round` 的编排：流、会话、指针契约与销毁

`run_round` 里有几条只有读源码才看得见的契约：

- **流依赖**：`constructor.wait_stream(current)`（`:169`）——BufferBuffer 在非默认流上构造；
  producer 端 `producer.wait_event(ready)`（`:229`，等 buffer 就绪）+ `wait_event(inputs_ready)`
  （`:230`，等参考数据构造完）后写输入；launch 端 `launch.wait_event(produced)`（`:236`）后才发集合；
  consumer 端 `handle.current_stream_wait(release_handle=...)`（`:270`）后取结果。三条流通过 event
  形成一个**跨流依赖链**。
- **指针契约**（`:277-282`）：非会话模式下，结果张量的 `data_ptr()` 必须**恰好**等于 arena 里 payload
  的地址（`reduce_scatter` 再加 `group.rank()*nbytes`）——坐实集合算子是**原地**在对称存储上做，
  而不是另开输出缓冲。`all_gather` 的 ptr 检查用的是 **arena 的 payload 地址**（`:279`），因为
  all_gather 的 dst 就是 arena 本身。
- **SM 校验**（`:273-276`）：`all_gather` 之外的 `reduction_sms == handle.num_sms`，而 `num_sms=None`
  时用 `buffer.get_theoretical_num_sms(op, group)`（`:275`，第 63 篇的解析式 roofline）——即
  「自动 SM 数」必须与理论模型一致。
- **快照时机**（`:283-284`）：`wait()` 之后**立即** `clone` 快照，注释明说「deferred CPU checks
  cannot hide missing dependencies」——防止把同步点藏进后续 CPU 操作。
- **barrier 变体**（`:296-302`）：每 7 步调 `buffer.barrier`，并轮换
  `wait_comm_stream` / `with_cpu_sync` / `sequential` 三个开关（第 53 篇）。
- **销毁**（`:304-314`）：清空 handles 后（若 `explicit`）`buffer.destroy()`，
  再 `weakref.ref(buffer)` + `del` + `gc.collect()` 断言 `buffer_ref() is None`——即**buffer 不能被
  测试自身意外持有**（否则记录流/句柄泄漏）。

---

## 9. `test_perf.py`：profile 矩阵与「慢 rank 中位数」

`test_perf` 的目标是「DeepEP vs 多档 NCCL 调优」：

- **NCCL profile 表**（`:36-48`）：`auto`/`cta32`/`cta64`/`ring`/`tree`/`nvls`/`nvls-tree`，每个是一组
  `NCCL_*` 环境变量；`NCCL_TUNING_VARS`（`:45-48`）列出切换 profile 前要清空的 8 个变量。
  `run_profiles`（`:218-249`）**总是先跑 auto**，再逐个 profile 起新 `mp.spawn`（注释「Fresh workers
  inherit this profile before importing PyTorch」——NCCL 调优必须在 import torch 前设好）。
- **coalescing**（`:76-84`）：`nccl_collective` 用 `dist._coalescing_manager` 把多个小集合**合并**成
  一批，baseline 取 loop/coalesced 两者最小值（`:160-164`）。
- **计时 `measure`**（`:51-73`）：warmup → 每个样本「flush L2 → barrier → `cuda._sleep` → start →
  fn → end」→ `all_reduce(times, MAX)`（取**每个样本全网最慢 rank 的时长**）→ **中位数**。
  取 max 是因为集合算子的完成时间由最慢 rank 决定，取中位数是抗离群。

---

## 10. H100(sm90) 真机实测

`harness/exp_deepep_bucket_test_oracle.{sh,py,out.txt}`：按文件路径 exec 两个真实测试模块（stub
`deep_ep` 绕过 `check_nccl_so`），在 8×H100 上用真实 NCCL 驱动纯逻辑部分。**33 PASS / 0 FAIL**：

| 组 | 内容 | 结果 |
|---|---|---|
| A | 源码加载：`test_mixed.py` sha256 `744379f9…`、`test_perf.py` `a78f9de8…`；真实常量 | OK |
| B | `supported` 8 物理域 × 4 op × 2 精度 = 64 组 vs 独立 oracle | 64/64 |
| C | `make_sizes`（mixed）3 组配置，尺寸与 oracle 逐项相等且全为 `quantum` 倍数 | 3/3 |
| D | `make_groups`（真机 `dist.new_group`）nvl∈{4,8,2} 的拓扑与 oracle 逐组相等 | 3/3 |
| E | `nccl_reference` 4 op × scale∈{1,0,-0.5} vs 独立求和期望；`rs_ag` 不变量 | 11/11 |
| F | `make_inputs` 形状/guard + `verify` 的 guard 抓错、exact 1-ulp、容差放行 | 8/8 |
| G | perf：`NCCL_CONFIGS`/`NCCL_TUNING_VARS`/`get_nccl_sm_budget`/ragged `make_sizes`/`get_skip_reason` | 6/6 |
| H | 真实 `measure`（8 卡 all_reduce 4MiB 中位时长） | 1/1 |

实测要点：

- **Part D**：`nvl=8` 时每 rank 属 2 组（world + half），oracle 全体 3 组；`nvl=4` 时每 rank 属 3 组
  （尺寸 `[2,4,8]`），全体 7 组（含 4 个 rail）；`nvl=2` 时每 rank 属 4 组（`[2,4,4,8]`），全体 9 组。
- **Part E**：`all_reduce`/`rs_ag`/`reduce_scatter` 对 `scale∈{1.0,0.0,-0.5}` 输出都与独立期望**逐位**
  相等；`all_gather` 输出 512 元素；`all_gather(reduce_scatter(x)) == all_reduce(x)` **逐位**成立。
- **Part F**：`verify` 对「guard 被改一个元素」「exact 模式下 1-ulp 差」都抛 `AssertionError`，对
  「exact=False 且 2e-4 容差内」放行——坐实两个判据分支都在工作。
- **Part G**：`get_nccl_sm_budget('cta64')=64`；ragged `make_sizes` 与 oracle 逐项相等（含 shuffle）；
  `get_skip_reason` 全域与 oracle 一致。
- **环境发现**：本机 torch 缺 `reduce_scatter_single`/`all_gather_single`（DeepEP 测试用新名），
  实验里用 `*_tensor` 兼容别名驱动；这是继第 86 篇（NCCL 版本）之后又一条「测试所需环境」的记录。

端到端 `BucketBuffer`（`run_round`/`test`/`all_reduce`/`reduce_scatter`/`all_gather`/`barrier`/
`session`/`destroy` 的 kernel 侧）需要 NCCL GIN 与对称窗口，**仅分析**——但它们的**规范**（门控、
尺寸、拓扑、参考、判据、计时）已在本篇逐条落到可执行代码。

---

## 11. 与既有篇章的关系

- 第 54 篇：bucket 三类算子 × 三物理域 + copy-engine（本篇讲它们**怎么被验收**）。
- 第 62 篇：`all_reduce`/`reduce_scatter` 三后端与 bf16 随机舍入（本篇的 `supported` bf16 门控、
  `rs_ag` 不变量的理论依据）。
- 第 63 篇：SM/QP 自动选择模型（本篇 `run_round:275` 的 `get_theoretical_num_sms` 校验，perf 的
  `get_nccl_sm_budget` 对照）。
- 第 65 篇：通信-计算重叠的 Python 调度地基（本篇 `run_round` 三条流的独立交错是它的测试侧镜像）。
- 第 83 篇：`BucketSession`（本篇 `run_round:165/239` 每 3 轮用它做临时 scratch）。
- 第 75 篇：DeepEP EP 测试即规范（本篇是 bucket 侧的对称物）。

**一句话**：`test_mixed.py` 用「能力门控 + 尺寸量化 + 拓扑划分 + NCCL 参考 + guard/逐位判据 +
三流交错」把 bucket 混合集合的正确性编码成可执行规范；`test_perf.py` 用「profile 矩阵 + coalescing +
ragged + 慢 rank 中位数」定义它的基准；二者共同回答「bucket 算什么对 / 跟谁比」。
