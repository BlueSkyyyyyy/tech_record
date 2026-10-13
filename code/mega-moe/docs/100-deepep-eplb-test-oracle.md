# DeepEP EPLB 冗余权重的验收测试即正确性规范（`tests/ep/test_prefetch_weights.py` + `test_reduce_grads.py`）

> 第 53 篇（P43-1）把 DeepEP 里与**动态专家复制（EPLB / redundant experts）**配套的两个通信原语
> ——`lb_prefetch_weights`（前向：把本 rank 的专家权重推到各 peer 的冗余槽）与
> `lb_reduce_grads`（反向：把各 peer 冗余副本的 FP32 梯度加回 owner）——从 host 编排到
> 负载均衡切块讲透了，也把它们和 MoE 反向的四个 grouped GEMM 接了起来。但那一篇只在
> §5/§6 里**引用**了上游的两个验收测试（`tests/ep/test_prefetch_weights.py` 250 行、
> `tests/ep/test_reduce_grads.py` 256 行）来佐证 reduce-add 的顺序无关性，从没把这两份
> 「什么叫对」的**唯一规范**单独拆过。本篇补上。
>
> **可跑性**：两个 kernel 走 NCCL 设备 API（`ncclDevComm` / `ncclGin`），本机系统 NCCL 2.29.7
> 的头不提供该 API（与第 7/53/86 篇同源「阻塞」），所以**设备路径端到端仅分析**。但这两份测试把
> 几乎全部判据都编码在**纯 Python / 纯张量**里——唯一 key 的确定性 ramp、映射表的 own-expert
> 排除与空槽语义、哨兵回填、原子加的顺序无关性。这些在 H100(sm90) 上以**按文件路径 ast 抽取
> 的真实函数**原样运行、独立复刻对拍。实验：
> `harness/exp_deepep_eplb_test_oracle.{py,sh,out.txt}`，源码 commit `93eb6eb`。

---

## 1. 为什么冗余专家需要这对原语（测试在验什么）

标准 EP 里每个 rank 持 `num_local_experts = E / num_ranks` 个专家，token 按 top-k 路由到专家
所在 rank。专家热度极不均匀，于是把热门专家的**副本**放到同域其他 rank 的空闲槽（EPLB）。这产生
两条跨 rank 数据流，正是两个测试各自验证的对象：

| 方向 | 原语 | 拷什么 | 期望终态 |
|------|------|--------|----------|
| 前向 | `lb_prefetch_weights` | 主人的**权重（+两组量化 SF）** → 副本槽 | 副本槽逐字节等于主人的 blob；空槽保持哨兵 |
| 反向 | `lb_reduce_grads` | 各副本的 **FP32 梯度** → 主人 | 主人梯度 = seed + 所有指向它的副本梯度（元素级原子加） |

两份测试的**结构几乎对称**：定义一种「确定性、可区分」的 blob，定义一张**每 rank 相同**的冗余
映射表，填种/哨兵 → 调原语 → 逐字节/逐元素检查 → 最后跑一遍 perf 并算带宽账目。真正的信息量在
「blob 怎么造、映射怎么造、判据为什么必须这么硬」这三件事上。

---

## 2. `test_prefetch_weights.py`：blob 与映射表

### 2.1 哨兵与载荷清单（`:16-24`）

```python
SENTINEL = 0xa5                                   # :16
PAYLOAD_WEIGHTS, PAYLOAD_SF0, PAYLOAD_SF1 = range(3)   # :19
PAYLOAD_NAMES = ('weights', 'sf0', 'sf1')         # :20
NUM_PAYLOADS = len(PAYLOAD_NAMES)                 # :21
NUM_BLOB_VALUES = (1 << 8) - 1                    # :24  = 255
```

一个专家要推**三种载荷**：权重本体 + 两组 scale factor（SF0/SF1 对应 FP8 权重的两种量化粒度）。
`SENTINEL = 0xa5` 是「未分配槽」的初值——**测试注释（`:15`）点破了它的用途**：填进每一个冗余槽
之后，`expert == -1` 的槽必须**仍然是 0xa5**，否则说明 kernel 写串了地址（strayed writes）。
`NUM_BLOB_VALUES = 255` 是 blob ramp 的周期，见下。

### 2.2 `make_blob`：唯一 key 的 ramp（`:45-53`）

```python
def make_blob(global_expert_idx, num_bytes, payload_idx) -> torch.Tensor:
    key = global_expert_idx * NUM_PAYLOADS + payload_idx          # :51
    idx = torch.arange(num_bytes, dtype=torch.int64, device='cuda') + key   # :52
    return (idx % NUM_BLOB_VALUES).to(torch.uint8)               # :53
```

**为什么不用随机数**（注释 `:46-50`）：它必须在**每一张 GPU 上逐位相同**，而 RNG 的发射配置
依赖设备属性；而且它要**纯算术**。`key = e*3 + p` 让 `(专家, 载荷)` 对唯一映射到一个相位——一个
从错误专家或推到错误载荷的 blob，会在**每一个字节**上都不同，而不是碰巧错几个字节。这个「全字节
失配」的性质就是判据硬度的来源。

**H100 实测**（Part B）：确定性、dtype/shape、逐字复刻 `(arange(n)+key)%255` 全部吻合；错误专家
（`e=6` vs `e=5`）与错误载荷（`p=2` vs `p=1`）的失配字节数**都恰等于 n**（4096/4096）。

### 2.3 `make_redundancy_mapping`：每 rank 相同的映射表（`:56-92`）

返回 `[num_ranks, num_redundant_experts]` 的 int32 表，**LSA 域内每个 rank 都相同**。行 `l` 是域内
第 `l` 个 rank，条目是域内专家 id `[0, num_experts)`，`-1` = 空槽。两个 pattern：

- **`random`**（`:81-90`）：从**远端专家**里采样，再 `+= (>= own_block_start) * num_local_experts`
  把它**跳过本 rank 的块**（`:87`），最后按 `empty_ratio` 随机置 `-1`（`:88-90`）。
- **`ring`**（`:75-80`）：rank `r` 的每一个冗余槽都取 rank `r+1` 的专家（注释 `:78`：偏移永不
  `≡0 (mod num_ranks)`，所以不会自借）。这是**基准 pattern 而非真实 pattern**（注释 `:75-80`）：
  它让每个 rank 只发一个 peer、只收一个 peer，去掉随机 pattern 的发送不均衡与接收碰撞，把「离
  NVLink 峰值差多少」里「流量形态」与「kernel」两部分分离开。

`:93-94` 是**结构不变量**的断言：`owners != 行号` 对所有非空槽成立——**一个 rank 绝不为自己的
本地专家持有冗余副本**。这条不变量在反向侧会变成「归约不会重复计 seed」，见 §3.3。

**H100 实测**（Part C）：`[8,8]` int32、own-expert 全排除、专家 id 域内、`empty_ratio` 实测 0.2656
（目标 0.25，8×8 采样误差内）、同 seed 确定 / 换 seed 不同、`ring` 的 owner 恰为 `(r+1) % NR` 且
无空槽——全过。

### 2.4 `check`：哨兵 vs blob 逐字节（`:95-108`）

```python
for redundant_expert_idx in range(redundant_expert_weights.size(0)):
    expert_idx = int(redundancy_mapping[rank_idx, redundant_expert_idx].item())
    if expert_idx < 0:
        assert (redundant_expert_weights[redundant_expert_idx] == SENTINEL).all()   # :104
    else:
        assert torch.equal(redundant_expert_weights[redundant_expert_idx],
                           make_blob(expert_idx, num_bytes, payload_idx))           # :106
```

两条分支：空槽必须**全哨兵**、分配槽必须**逐字节等于该专家该载荷的 blob**。判据用 `torch.equal`，
**不是容差比较**——因为 blob 是纯算术的确定值，任何偏差（错专家、错载荷、错偏移、字节错位）都会
被逐位抓到。

**H100 实测**（Part D）：正确内容通过（含 `-1` 槽保哨兵）；三个负样本——错专家、未分配槽被写入、
错载荷——**全部被 `AssertionError` 抓到**。

### 2.5 测试主体（`:111-227`）与全量 sweep

正确性部分（`:176-202`）对 `PAYLOAD_SWEEP`（`:27-38`，10 个 `(weights, sf0, sf1)` 字节三元组）
逐载荷 × `with_sf ∈ {True, False}` × `num_rounds` 轮：

1. **先填哨兵再同步再 barrier**（`:183-187`）——注释 `:183` 点明「哨兵必须先对所有 peer 可见，
   才能有人开始推」，否则 peer 可能读到旧值；
2. 隔轮把**扁平张量换成 3D/4D 视图**（`reshape_weights`，`:41-42`；调用 `:190-193`），尾随形状逐轮
   变化但**每专家字节数不变**——这是在验「`redundant_expert_weights` 只要 `[num_redundant_experts,
   *shape]` 连续、字节对齐即可」的 host 契约（`ep.py:865-873` 的 docstring）；
3. `with_sf=False` 时**只传 `[:1]`**（`:169-170`），于是**未被请求的载荷槽必须仍是哨兵**（`:200-202`）。

`PAYLOAD_SWEEP` 的每一行都被选来打到一条边界（注释 `:28-37`）：4 MiB 让 warp 缓冲跨多个 barrier
相位复用、48 KiB 是 12 KiB chunk 容量的整数倍、12320/12256 分别落在 chunk 两侧带短尾、20480 让
每个载荷都落在部分 chunk、1152/544/288 全部走 TMA、96/64/32 让 chunk 数少于 grid 的 warp 数。

**H100 实测**（Part G）：10 载荷 × 2 轮 × 8 rank 的**全槽 push 仿真** 920 槽 0 不符；每载荷满足
`payload[i] % 32 == 0`（`deep_ep` 的 `kNumTMAAlignmentBytes`，`compiled.cuh:67`，经
`python_api.cpp:25` 转出）；blob key 预算 `num_experts*3 ≤ 255`（`:131-132` 的断言，`64*3=192`）。

### 2.6 带宽账目（`:204-222`）

perf 段先统计「每个 sender rank 被请求了多少个冗余专家」（`:207-208`），再按
`num_bytes_per_redundant_expert = sum(perf_payload)`（`:209`）算两个数：

- **this rank** 实际推的字节 = `redundant_experts_per_sender[rank] * bpe`；
- **critical path** = `max(redundant_experts_per_sender) * bpe`（`:211`）——最忙的那个 sender 的
  字节数，因为所有 rank 必须等最慢的 peer 完成。

最后除以 `get_nvlink_gbs()`（`envs.py:207-234`，`nvidia-smi nvlink -s` 各 link 带宽求和 ×0.9）
得「占 NVLink 百分之几」。**H100 实测**（Part G，`random` seed 0）：每 rank 推送的冗余专家数
`[2,5,6,8,9,10,4,3]`，每冗余专家 67.13 MB，critical path 671.3 MB；本机 NVLink 上限实测
**430.3 GB/s**，故 critical path 的**物理下限约 1560 us**（100% 占用时）——这就是 `random` 的
不均衡（最忙 10 个、最闲 2 个，差 5×）为什么会让 GB/s 难读，也是 `ring` pattern 存在的理由。

---

## 3. `test_reduce_grads.py`：反向的对称规范

### 3.1 `make_pattern`：顺序无关的精确加（`:23-40`）

```python
NUM_PATTERN_VALUES = (1 << 12) - 1                                    # :17 = 4095

def make_pattern(hidden, key):
    idx = torch.arange(hidden, dtype=torch.int64, device='cuda') + key
    return ((idx % NUM_PATTERN_VALUES) - NUM_PATTERN_VALUES // 2).to(torch.float32)   # :39-40
```

与 prefetch 的 `make_blob` 同构，但**值是小整数**（`(0..4094) - 2047`）。注释 `:30-34` 给出**核心
理由**：`lb_reduce_grads` 用 `cp.reduce.async.bulk`——**元素级原子加**，多个 borrower 落地的顺序
不固定；值足够小则 fp32 下累加**精确**，于是期望值**与顺序无关**，判据可以用 `torch.equal` 而
**不需要容差**。这把「kernel 是不是加对了」与「浮点求和顺序」两件事解耦。

**H100 实测**（Part E）：dtype/逐字复刻/取值范围全对；5 个 pattern 三种加序 `torch.equal` 逐位
相等；周期 4095 不被 12 KiB chunk（3072 个 fp32）整除，故相邻 chunk 内容必不同（注释 `:16` 的
用途）——全部吻合。

### 3.2 两族 key：expert 与 redundant 永不相交（`:43-51`）

```python
def expert_grad_key(expert_idx): return expert_idx                          # :46  >= 0
def redundant_expert_grad_key(rank_idx, redundant_expert_idx, num_redundant_experts):
    return -1 - (rank_idx * num_redundant_experts + redundant_expert_idx)   # :51  < 0
```

**owner 的初始梯度用非负 key**，**每个副本的梯度用负 key**——两族永不碰撞。这样当 `seed + Σ副本`
恰好等于期望值时，可以确信加的是「正确那份」而非某个同值巧合。`redundant_expert_grad_key` 在
`(rank_idx, redundant_expert_idx)` 上唯一。

### 3.3 `expected_expert_grads`：归约期望值（`:98-114`）

```python
expected = stack([make_pattern(hidden, expert_grad_key(rank*num_local_experts + le)) for le in range(nle)])
for remote_rank_idx in range(num_ranks):
    if remote_rank_idx == rank_idx: continue                    # :107 绝不从自己那行 pull
    for redundant_expert_idx in range(num_redundant_experts):
        expert_idx = mapping[remote_rank_idx, redundant_expert_idx]
        if expert_idx >= 0 and expert_idx // nle == rank_idx:   # :111 这个副本属于我
            expected[expert_idx % nle] += make_pattern(hidden, redundant_expert_grad_key(...))
```

期望 = **自己的 seed** + 所有**指向自己**的副本梯度。`:107` 的 `continue` 对应 §2.3 的 own-expert
排除：映射表保证没有任何 rank 会从自己那行 pull，所以 seed 只出现在期望里一次、不会被重复计。
`:111` 的 `expert_idx // nle == rank_idx` 是「这个冗余槽的主人是不是本 rank」——域内专家 id 的高位
就是 owner 的 rank。

**H100 实测**（Part F）：真实 `expected_expert_grads` 与**独立复刻**逐位相等；本 rank 被拉的副本数
= 8（>0）；own-expert 排除坐实。

### 3.4 `seed_buffers` 与判据（`:163-204`）

正确性段对 `CORRECTNESS_HIDDENS`（`:20`，11 个 hidden：1M/256K/16392/6144/3080/3072/3064/5120/
1536/384/24）逐 hidden × 轮：

- **每次都要重填 seed**（`seed_buffers`，`:163-169`；注释 `:164`）：`expert_grads` 是**累加**目标，
  上轮的结果必须清掉；
- 归约后逐本地专家 `torch.equal(expert_grads[le], expected[le])`（`:191-196`），失败时打印**多少个
  元素错**；
- **额外再断言 `redundant_expert_grads` 一个元素都没被改**（`:200-204`）。注释 `:198-199` 点破：如果
  把 TMA 的源/目的弄反，错会落在这条断言上而不是上一条——所以值得单独测。

**H100 实测**（Part G）：11 hidden × 2 轮 × 8 rank 的 pull 期望值 176 组 0 不符；每 hidden 满足
`hidden*4 % 32 == 0`；pattern key 预算 `num_experts + num_ranks*num_redundant_experts < 4095`
（`:136-137` 的断言，`64 + 64 = 128`）。

### 3.5 带宽账目与 prefetch 的对偶（`:206-233`）

reduce 段统计的是**每个 owner rank 被多少副本拉**（`:210-211`），与 prefetch 段统计的「每个 sender
被请求多少」**是同一个集合、只是从另一端读**。这解释了为什么 prefetch 与 reduce 共用同一张
`redundancy_mapping`（`ep.py:894` 的 docstring：「the mirror of `lb_prefetch_weights`: use the
same redundancy mapping to return gradients」）。

**H100 实测**（Part G）：用同一 mapping，`[2,5,6,8,9,10,4,3]` 两段**完全一致**；reduce 的 perf
hidden（64 MiB / 4 = 16M 个 fp32）下每冗余专家 67.11 MB、critical path 671.1 MB。

---

## 4. host 契约：测试断言背后的 API（`deep_ep/buffers/ep.py`）

两份测试调用的 host 入口：

- `lb_get_theoretical_num_sms:846-852`：把 `num_sms=0` 解释成「按带宽预算估一个 SM 数」——
  `align(ceil(max(nvlink/sm_read, nvlink/sm_write)), 2)`，`sm_read_gbs=180` / `sm_write_gbs=45`
  （`envs.py:193-204`）。因为要**同时**从 HBM 读权重、往 NVLink 写，取两个方向的较大者。
- `lb_prefetch_weights:854-883`：把单张量包成 list、`num_sms=0` 走上面那个估计、再 `align(num_sms, 2)`
  转发给 `self.runtime.lb_prefetch_weights`。docstring（`:865-873`）写明三条契约：**collective**
  （LSA 域内每个 rank 都要调）、`redundant_expert_weights` 必须**连续且落在本 buffer 的 LB 区**、
  `expert_weights` 的**每专家字节数必须是 `get_num_tma_alignment()` 的倍数**（尾随形状可不同）。
- `lb_reduce_grads:886-905`：镜像入口，docstring `:895` 强调**是加不是覆盖**，所以调用前要先 seed。
- `BufferAllocator`（`deep_ep/buffers/allocator.py:10-35`）与
  `EPBuffer(..., lb_allocation_plan_or_num_bytes=plan, explicitly_destroy=True)`（`ep.py:242-315`）：
  两份测试都用「规划期分配」把冗余专家池作为**对称 buffer 的 LB 区视图**一次性规划好（`plan.allocate`
  `:136-137` / `:140-141`），测试末尾手动 `buffer.destroy()`（`:224-227` / `:235-238`）。

注意测试在 `import deep_ep` **之前**先 pop 掉 `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS`/
`NCCL_NVLS_ENABLE`（两份文件 `:1-6`）——避免环境变量改变 NVLS/通道数从而改变 kernel 行为，保证
可复现。

---

## 5. 结论

- 这两份测试是 EPLB 两个原语的**可执行正确性规范**：一条 **唯一 key 的确定性 ramp** 让任何
  错专家/错载荷/错偏移都在**每个字节**上暴露；一张 **own-expert 排除**的映射表同时定义了前向
  「往哪推」与反向「从哪拉」；反向的**小整数精确 fp32** 让原子实在 `torch.equal` 下顺序无关。
- 三条判据维度彼此独立：**哨兵分支**（空槽不许被写）、**未请求载荷分支**（`with_sf=False` 时另两槽
  必须是哨兵）、**反向的源池不被写**（TMA 源/目的弄反的落点）。
- **H100 真机 37/37 ALL PASS**（`harness/exp_deepep_eplb_test_oracle.{py,sh,out.txt}`）：认证两份源码
  sha256、原样抽取真实函数运行、独立复刻 `make_blob`/`make_pattern`/`make_redundancy_mapping`/
  `expected_expert_grads` 逐位对拍、三个负样本被断言抓到、全量 sweep 仿真 0 不符、prefetch 与
  reduce 的带宽关键路径一致（本机 NVLink 上限 430.3 GB/s，critical path 下限 1560 us）。
- **设备路径（`lb_prefetch_weights` / `lb_reduce_grads` 的 NCCL GIN kernel）端到端仅分析**——与
  ROADMAP「阻塞」同源，修好 ≥2.32.3 的 NCCL 环境后可做 8×H100 端到端实测。
