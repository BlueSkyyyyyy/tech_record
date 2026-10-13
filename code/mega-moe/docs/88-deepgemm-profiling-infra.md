# 88. DeepGEMM 自带的 Mega MoE profiling 基础设施与自动生成的「覆盖面清单」

> 本文对应路线图 **P78-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 88 篇。
>
> 前面第 3/9/10/11 篇把 Mega MoE 主 kernel 逐段拆开，第 15 篇用 `ncu` 量了反向 SM90 kernel 的
> SpeedOfLight / 定向 metric。但一直没回答一个很工程的问题：**DeepGEMM 官方自己是怎么 profile
> 那颗 SM100 Mega MoE kernel 的？** 它没有用「一句 `ncu --set full python xxx`」，而是自带了一套
> **多进程 NCU 驱动 + PM-sampling 报告后处理**的基础设施。本篇把这套设施逐行讲清：
>
> - `scripts/run_ncu_mega_moe.sh`（89 行）——多进程 NCU launch driver；
> - `tests/test_mega_moe.py`（467 行）的 `--ncu-profile-only` / `--local-rank-idx` / `--dump-profile-traces`
>   与**性能账目公式**；
> - `scripts/quick_plot_pm.py`（448 行）——把 `.ncu-rep` 的 PM-sampling 时间序列画成图；
> - 顺带补上两份**自动生成/索引的「覆盖面清单」**：`scripts/generate_pyi.py` → `stubs/_C.pyi`
>   （**Python API 面**）与 `csrc/indexing/main.cu` + CMake（**CUDA kernel 面**）。
>
> **可跑性**：`run_ncu_mega_moe.sh` 的目标 kernel 是 **SM100 专属 + 依赖 DeepEP**，本机 H100(sm90)
> 端到端**跑不了**（仅分析）；但本篇的三类实验全部是**架构无关的真机验证**——
> ① 用**桩 `ncu`/`python` 拦截真实脚本**、捕获它构造的原始 argv；② 在 H100 上生成**真实 `.ncu-rep`
> 并跑 `ncu --import --page raw --csv`**，验证后处理脚本的取址格式；③ 直接运行**真实的
> `generate_pyi.py` 与真实 `test_mega_moe.py` 代码块**。
>
> 源码坐标：`~/github/DeepGEMM`，本地 commit `057ca59`（v2.8.1）。实验：
> `harness/exp_megamoe_profiling.{sh,py,out.txt}`。

与其它篇的关系：第 10 篇讲主 kernel 的 MMA/epilogue、第 11 篇讲 dispatch/combine 与 NVLink 访存、
第 15 篇讲反向 kernel 的 `ncu` 归因、第 41 篇讲 `test_mega_moe.py` 的**正确性**判据、第 84 篇讲
Python API 面的**人工枚举**。本篇补的是「**官方的性能验证与工具链面**」：怎么 profile、怎么把报告变成图、
以及 API/kernel 的「覆盖面清单」怎么被自动生成——并暴露其中**一处不可复现的工具 bug**。

---

## 1. 为什么 Mega MoE 的 profile 要另起一套

Mega MoE 是一颗 **persistent kernel**（`grid = num_sms`，见第 3 篇），单次 launch 几百微秒，
把 dispatch、两次 grouped GEMM、SwiGLU、combine 全融进一段「通信—计算重叠」的流水线里。
用常规做法有四个麻烦：

1. **不能只测一次**：kernel 有长尾（wave quantization、ring buffer 排空），要看**执行期间随时间变化的
   SM/L2/DRAM/NVLink 占用**，而不是一个平均标量。这正是 NCU 的 **PM Sampling**（`PmSampling` section）
   提供的：以固定周期采样硬件 PM 计数器，输出**时间序列**。
2. **要精确点名 kernel**：一个进程里除了 `sm100_fp8_fp4_mega_moe_impl` 还有一堆辅助 kernel，
   必须用 `--kernel-name` 只抓目标，否则报告里全是噪声。
3. **要多进程一起被抓**：Mega MoE 的输入来自 EP dispatch，需要 8 个 rank 同时跑。
   `ncu` 默认只 wrap 一个进程，所以要**每 rank 一个 `ncu`**，并让它们彼此能通信（TCP communicator）。
4. **要有源码行号归因**：PM sampling 的价值一半在 `SourceCounters`——把 stall 归因到 SASS 行，
   这要求 JIT 编出的 cubin 带 lineinfo（`DG_JIT_WITH_LINEINFO=1`）。

这四点决定了 `run_ncu_mega_moe.sh` 的形状。

---

## 2. `scripts/run_ncu_mega_moe.sh` 逐段

### 2.1 参数解析（`:8-46`）

脚本自己写了一个 `for ((arg_idx = 1; ...))` 的循环（而不是 `getopt`），把参数分成两拨：

- `--num-processes N` / `--num-processes=N`（`:15-24`）：同时记进 `num_processes`，并**塞进
  `python_args`**——因为下游 `test_mega_moe.py` 也要用它（见 §3）。
- `-o DIR` / `--output DIR`（`:28-36`）：只作为 NCU 报告的输出目录 `output_dir`，**不进** `python_args`。
- `-h` / `--help`（`:25-28`）：打印用法后 `exit 0`。
- 其余一律 `python_args+=("$arg")`（`:43`）：透传给 `test_mega_moe.py`（`--hidden`、`--num-experts` 等）。

默认 `num_processes=8`、`output_dir=work`（`:4-5`）。最后 `mkdir -p $output_dir`（`:46`）。

### 2.2 JIT 预热 + lineinfo（`:48-53`）

```bash
export DG_JIT_WITH_LINEINFO=1   # :48  for source counters
echo "Warm up JIT cache"
python tests/test_mega_moe.py --ncu-profile-only "${python_args[@]}"   # :51
sleep 2                          # :53
```

三行各有讲究：

- `DG_JIT_WITH_LINEINFO=1` 让 DeepJIT 编译时带行号信息，否则 `SourceCounters`/PM sampling 里
  的 stall 只能落到 SASS 指令、落不到源码行。
- **先跑一遍不带 NCU 的**（`:51`）：把 DeepJIT 的编译缓存填好。NCU 用 `--replay-mode application`
  **整进程回放**，如果第一次编译发生在被 profile 的那次回放里，会把编译时间算进报告、还会重复编译。
  预热一次后，被 profile 的进程直接命中缓存。
- `sleep 2`（`:53`）：等上一步的进程/缓存落盘（DeepJIT 原子发布，见第 23 篇）稳定。

### 2.3 NCU 参数集（`:55-74`）

```bash
ncu_args=(
    --config-file off                        # :56 忽略 ~/.ncu 配置，保证可复现
    --force-overwrite                        # :57 覆盖同名 .ncu-rep
    --kernel-name sm100_fp8_fp4_mega_moe_impl # :58 只抓 Mega MoE 主 kernel
    --import-source yes                      # :59 把源码嵌入报告
    --replay-mode application                # :60 整进程回放（多进程通信必须）
    --section PmSampling                     # :61 PM 采样：硬件计数器时间序列
    --section SourceCounters                 # :62 源码级 stall 归因
    --rule LocalMemoryUsage                  # :63 本地内存(溢出)规则
    --launch-skip 0 --launch-count 1         # :64-65 只抓第 1 次 launch
    --lockstep-kernel-launch                 # :66 多进程 kernel 对齐
    --communicator tcp                       # :67 多进程通信走 TCP
    --clock-control none                     # :68 不改频（否则时间序列会失真）
    --pm-sampling-interval 1000              # :69 每 1000 周期采样一次
    --pm-sampling-max-passes 1               # :70 采样遍数
    --disable-pm-warp-sampling               # :71 关掉 warp 级 PM（降噪/降重放次数）
    --communicator-tcp-num-peers "$num_processes"  # :72 peer 数=rank 数
    --kill yes                               # :73 结束后杀掉应用
    --app-replay-buffer memory               # :74 回放缓冲放显存
)
```

几个关键点：

- `--kernel-name sm100_fp8_fp4_mega_moe_impl` 用的是 JIT **实例化后的模板名**（第 23 篇：`_impl` 后缀），
  不是 Python 里的 `fp8_fp4_mega_moe`。
- `--replay-mode application` + `--communicator tcp` + `--communicator-tcp-num-peers N` 是**多进程
  profile 的三件套**：NCU 要让 N 个被 wrap 的进程在每次回放时同步，否则跨 rank 的 dispatch/combine
  数值会错、时间序列会错位。
- `--clock-control none`：NCU 默认会锁频以保证可复现，但 PM sampling 要的是**真实时间轴**，
  锁频反而引入伪影，所以显式关掉。
- `--disable-pm-warp-sampling`：warp 级 PM 需要更多 replay pass，而对「SM/L2/DRAM/NVLink 的粗粒度
  时间序列」没帮助，关掉更省时间。

### 2.4 每 rank 一个 NCU，后台并行（`:79-88`）

```bash
for ((i = 0; i < num_processes; ++i)); do
    ncu ${ncu_args[@]} -o "${output_dir%/}/mega-moe.$i" \
        python tests/test_mega_moe.py \
            --local-rank-idx=$i --ncu-profile-only "${python_args[@]}" &
done
wait
```

- **每 rank 一个 `ncu`**：`-o mega-moe.$i` 出 N 个报告；`&` 让 N 个 NCU 并行跑；`wait`（`:88`）汇总。
- `--local-rank-idx=$i`（配 `--ncu-profile-only`）让 `test_mega_moe.py` **以单进程、指定 local rank
  身份运行**——因为 `ncu` 只能 wrap 一个进程，不能 wrap `torch.multiprocessing.spawn`（见 §3.2）。

> **H100 实测（E1）**：本机端到端跑不了（SM100 + DeepEP），但脚本的**命令行构造**是纯 bash、架构无关。
> 我们在 `PATH` 前面放两个**桩** `ncu`/`python`（只把自己收到的 `$@` 打印到一个临时文件），然后**真跑
> 这个脚本**（`--num-processes 3 -o ... --foo bar`）。捕获到的**真实 argv** 完全印证上面的解读：
>
> ```
> python tests/test_mega_moe.py --ncu-profile-only --num-processes 3 --foo bar      # 预热（1 次）
> ncu --config-file off --force-overwrite --kernel-name sm100_fp8_fp4_mega_moe_impl
>     --import-source yes --replay-mode application --section PmSampling --section SourceCounters
>     --rule LocalMemoryUsage --launch-skip 0 --launch-count 1 --lockstep-kernel-launch
>     --communicator tcp --clock-control none --pm-sampling-interval 1000 --pm-sampling-max-passes 1
>     --disable-pm-warp-sampling --communicator-tcp-num-peers 3 --kill yes --app-replay-buffer memory
>     -o .../mega-moe.{0,1,2} python tests/test_mega_moe.py --local-rank-idx={0,1,2}
>     --ncu-profile-only --num-processes 3 --foo bar                                 # 每 rank 1 条
> ```
>
> 可见：`--communicator-tcp-num-peers` 精确等于 `--num-processes`；`--foo bar` 被透传进每条 python 调用；
> `-o` 按 rank 编号；预热那次没有 NCU 包装。见 `harness/exp_megamoe_profiling.out.txt` 的 E1 段。

---

## 3. `tests/test_mega_moe.py` 的 profile 模式与性能账目

### 3.1 `--ncu-profile-only`：只跑融合 kernel（`:233-241`）

```python
if args.ncu_profile_only:
    create_inputs()
    dist_print('Run fused kernel:', once_in_node=True)
    run_fused()
    dist_print(' > Done, exiting', once_in_node=True)
    dist.barrier(); buffer.destroy(); dist.destroy_process_group()
    return
```

它**跳过了正确性对拍**（baseline 需要 DeepEP + tilelang，见 `:243-260`）和 benchmark 循环，只做
「造输入 → 跑一次 `run_fused` → 退出」。`run_fused`（`:202-219`）本身是 `copy_inputs_to_buffer()`
+ 调 `fp8_fp4_mega_moe` / `bf16_mega_moe`，正是 NCU 要点的 kernel。

### 3.2 `--local-rank-idx`：单进程模式（`:454` / `:461-463`）

```python
parser.add_argument('--local-rank-idx', type=int, default=None,
                    help='Run as single process with this local rank (e.g. for NCU prof)')
...
if args.local_rank_idx is not None:
    test(args.local_rank_idx, args.num_processes, args)   # 单进程
else:
    torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
```

默认走 `mp.spawn`（每 rank 一个子进程）；给了 `--local-rank-idx` 就**不 spawn**，把当前进程当成
local rank `i` 直接跑 `test()`。这样 N 个 `ncu ... python ...` 平铺成 N 个进程，每个都被 NCU 独立 wrap，
并且靠 `init_dist`（`deep_gemm.utils.dist`）用环境变量/文件握手组成同一个通信组。

### 3.3 `--dump-profile-traces` + `bench_kineto`（`:362-363`）

```python
trace_path = None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json'
t_fused = bench_kineto(run_fused, 'mega_moe', barrier=barrier_fn, trace_path=trace_path)
```

这是**另一条正交的性能通道**：`bench_kineto`（第 30/40 篇）用 Kineto 计时，`trace_path` 非空时每个 rank
dump 一份 Chrome trace JSON，可以做「通信—计算重叠」的时间线分析（第 1/65 篇的 overlap 量化就是干这个的）。

### 3.4 性能账目公式（`:370-410`）

```python
num_routed_flops = 2 * num_recv_tokens * hidden * intermediate_hidden * 3   # :370
num_shared_flops = 2 * num_tokens * hidden * shared_intermediate_hidden * 3 # :371
num_nvlink_bytes = num_recv_tokens * hidden * 3                            # :398
t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12            # :401
approx_factor = safe_div(elapsed, elapsed - t_reduction)                    # :408
```

- **FLOP 的 `×3`**：Mega MoE 的 routed 部分是 L1（`2·M·H·(2I)` gate+up）+ L2（`2·M·I·H`），
  合计 `2·M·H·I·3`；`M = num_recv_tokens`（本地收到的 token 数）。
- **HBM 字节**（`:376-397`）：权重（按 `num_touched_experts` 个专家）× per-element 尺寸
  （bf16=2B / fp8=1B / **fp4=0.5B**）+ 激活读写 + 输出。
- **NVLink 字节** `= recv·hidden·3`：dispatch 远端 pull（1 份）+ combine 写回（2 份，因为 combine 要
  把 top-k 个贡献写回）。
- **`t_reduction`**：combine 的**串行归约**时间近似（`T·H·2·(1+topk) / 6.5e12`，6.5 TB/s 是经验带宽）。
  `approx_factor = elapsed/(elapsed−t_reduction)` 把「本来是串行归约」的那部分扣掉，**外推**成
  「如果归约也全重叠」的等价 TFLOPS/带宽——这就是打印里 `overlap: xxx TFLOPS` 的来源（`:404-417`）。

> **H100 实测（E3）**：把 `:366-401` 这段**真实代码**从 `test_mega_moe.py` 里按行抽出来、`exec` 执行
> （补上 `torch`/`gathered_topk_idx` 与一组样本参数），逐条对拍独立复算：`num_routed_flops`、
> `num_shared_flops`、`num_nvlink_bytes`、`t_reduction` 全部一致；fp4 路径 L1 权重字节 6.744e9 vs
> bf16 路径 2.663e10（≈4×，坐实 0.5B vs 2B）。见 `.out.txt` 的 E3 段（9 项全 PASS）。

---

## 4. `scripts/quick_plot_pm.py`：把 PM-sampling 报告画成图

这份脚本是 `run_ncu_mega_moe.sh` 的后半段——它读 NCU 生成的 `.ncu-rep`，把**认证过的一组指标**
从 PM-sampling 的原始数据里抠出来，画成一张「按类别分行的共享 x 轴」图。

### 4.1 指标清单 `CURATED_METRICS`（`:77-123`）

45 条指标，分 6 类（`CATEGORY_ORDER` `:47-54`）：**Overview**（launched/active blocks、CGA）、
**SM**（active cycles、IPC、各 pipe 吞吐、TMEM pipe）、**L1**（吞吐、sector、hit rate、wavefront）、
**L2**（吞吐、对 L1/GCC/DRAM 的分项、SysL2）、**DRAM**（频率、读/写吞吐）、**Interconnect**
（C2C/PCIe/NVLink TX/RX 吞吐与带宽）。

每条以 `MetricSpec(name, metric, kind, category)` 描述，`kind` 决定后缀（见 4.3）。指标名带
**TriageCompute 前缀**（如 `LTS.TriageCompute.lts__throughput`），那是因为它们在 GB100 的
**PM-sampling 域**下才能取到时间序列。注释（`:70-76`）明确记了几处「对着 `ncu --query-metrics --chip gb100`
修正过」的名字，比如 L2 对 GCC 请求的吞吐用 `lts__t_sector_throughput_srcunit_gcc`、NVLink RX 用
`NVLRX` 域。

### 4.2 chip 查询 + 别名回退（`:134-181`）

```python
def _query_available_metrics(chip):        # :134-153
    subprocess.run(['ncu','--query-metrics','--chip',chip], ...)
    # 取每行第一个 token，含 "__" 的即为 metric 名
def _metric_candidates(metric):            # :156-161
    # ['LTS.TriageCompute.lts__throughput', 'lts__throughput']  去掉 TriageCompute 前缀作为候选
def resolve_metric_specs(chip):            # :164-181
    # 对每条 spec：把 (metric + aliases) 的所有候选依次去前缀，取第一个在 available 里的
```

设计意图：**不同芯片/NCU 版本** 对同一个逻辑指标可能用不同的前缀或名字。脚本先 `ncu --query-metrics`
问出当前芯片**实际有哪些指标**，再对每条 spec 做「原名 → 去 TriageCompute 前缀 → 别名」的多候选匹配；
匹配不上的进 `missing`（打印出来），保证**跨芯片可移植**（图里少一条而不是崩掉）。

### 4.3 从 raw CSV 取址（`:126-131` / `:184-224`）

这是全脚本最容易读错的地方。它**不**用 NCU 的 GUI/JSON，而是：

```python
def _probe_metric_series(report, metric_name):    # :201-224
    rows = _run_csv_command(['ncu','--import',report,'--page','raw','--csv',
                             '--metrics',metric_name,'--print-metric-instances','values'], 60)
    if not rows or len(rows) < 3 or len(rows[0]) <= 11: return None
    header, units, row = rows[0], rows[1], rows[2]
    unit = units[11]; raw = row[11]
    return header[11], unit, _parse_metric_values(raw)
```

- `ncu --import <rep> --page raw --csv --metrics <m> --print-metric-instances values` 会输出一张 CSV：
  **前 11 列**是固定的 launch/device 元数据（`ID, Process ID, ..., Grid Size, Device, CC`），
  **第 12 列（下标 11）**才是本次请求的那个 metric 的**实例值**。
- 所以 `len(rows[0]) <= 11` 就说明「这个 metric 不在报告里」，直接返回 `None`（于是 `collect_metric_series`
  会对下一个后缀再试）。
- `--print-metric-instances values` 让**每个实例（每个 SM/slice/L2 分区…）的值**串成一条
  `"agg (v1 ; v2 ; v3 ; ...)"` 的字符串塞进第 12 列——这正是 PM-sampling 的**时间序列**（每个采样点一个值）。

`_parse_metric_values`（`:184-198`）解析三种形态：

| 输入 | 结果 |
|---|---|
| `"32 (32)"` | `(32.0,)`（单实例，非序列） |
| `"1.5 (1,000 ; 2,000 ; 3,000)"` | `(1000.0, 2000.0, 3000.0)`（聚合值 + 序列） |
| `"(10;20;30)"` | `(10.0, 20.0, 30.0)`（纯序列） |
| `"no data"` / `""` / 非数值 | `()` |

`collect_metric_series`（`:227-248`）只收下 **`len(values) > 1`** 的序列——也就是说
「只有单个实例、没有时间维」的 metric 会被刻意跳过（**PM sampling 要的是时间序列**）。

### 4.4 画图 `plot_pm`（`:274-417`）

把 45 条指标按 `CATEGORY_ORDER` 排序，画成 45 个共享 x 轴的子图（`GridSpec`），每行左侧标注
指标名、右侧标注峰值（`_format_value` 做 K/M/G/T 单位换算，`:251-265`），按类别换色并在类别切换处
打类别标签。百分比的 y 轴固定 0–100，其余按峰值自适应。默认存成 `<report>.pm_sampling.png`
（`:411-414`），`--interactive` 时开 Qt 窗口。

> **H100 实测（E2）**：本机没有 GB100 的 PM-sampling 数据，但我们**真的在 H100 上生成 `.ncu-rep`
> 并跑 `ncu --import --page raw --csv`**，验证了取址格式：
> ① 只请求一个**报告里存在**的 metric（`sm__maximum_warps_avg_per_active_cycle`）→ CSV 恰好 **12 列**，
> `header[11]` = 指标名、`unit[11]='warp'`、`row[11]="32 (32)"`，`_probe_metric_series` 复刻出
> `(name, 'warp', (32.0,))`；② 全量 raw（不指定 metric）→ **266 列**（11 + 采集到的 255 个 metric）；
> ③ 请求一个**不在报告里**的 metric → 11 列 → `_probe_metric_series` 返回 `None`（后缀回退生效）。
> 另外 `_parse_metric_values` 三种形态、`_metric_candidates` 去前缀、C2C 别名回退
> （只有 `ctx__throughput` 时回退成功）、单值序列被跳过、`plot_pm` 在 Agg 后端出 **47 KB PNG**——
> 全部通过。见 `.out.txt` 的 E2 段。

---

## 5. 附带：两份自动生成的「覆盖面清单」

`run_ncu_mega_moe.sh` / `quick_plot_pm.py` 是「性能验证」面。DeepGEMM 还有两份**自动生成/索引**的
「覆盖面清单」，恰好和第 84 篇的人工枚举互补。它们不是 SM100 专属，**H100 上纯 host 可跑**。

### 5.1 `scripts/generate_pyi.py` → `stubs/_C.pyi`（Python API 面）

`generate_pyi_file`（`:872-889`）三步：

1. `build_cpp_function_index(root)`（`:5-94`）：用正则扫所有 `.c/.h/.cpp/.hpp`，给每个**函数名**建一个
   「base name → 完整签名」的索引（去掉注释/`#` 行，跳过 `return/if/for/...` 关键字前缀）。
2. `extract_m_def_statements(root)`（`:151-223`）+ `parse_m_def_statement`（`:225-343`）：扫描所有
   `m.def(...)`，用一个自研的 `BracketTracker`（`:97-148`）**在顶层括号分割实参**（正确跳过
   `std::tuple<...>`、lambda 体里的逗号），取出 Python 名、C++ 函数名、以及 pybind 的 `py::arg(...)`
   默认值。
3. `extract_cpp_signature_details`（`:581-607`）→ `generate_pyi_function`（`:781-826`）：把 C++
   签名里的类型/参数名/默认值翻成 Python 注解（`cpp_type_to_python_type` `:608-698`、
   `cpp_default_to_python_default` `:723-779`），拼成 `.pyi`。

产出 `stubs/_C.pyi`（**612 行**）：

- **62 个 `def`**，正好对应源码里 62 个 `m.def`（第 84 篇：`_C` 顶层共 **76** 名 = 62 `m.def`
  + 10 `m.attr` 别名 + `Runtime`/`get_jit` + `epilogue`/`locality_domain` 两个子模块）。
- 其中 **14 个**是 `def xxx(*args, **kwargs) -> Any: ...` 的**回退**——那些是 **lambda 注册**的函数
  （`config.hpp:9-39` 的 `init/shutdown/set_num_sms/...` 与 `locality_domain`/`epilogue` 的部分），
  正则索引抓不到 lambda 的 C++ 签名，于是回退。**48 个**拿到了真实签名。
- **关键盲区**：pyi 只覆盖 `m.def`，**完全不含 `m.attr` 的 10 个别名**
  （`gemm.hpp:921-930`：`fp4_gemm_nt = fp8_fp4_gemm_nt` 等）。所以 `stubs/_C.pyi` 的 **62** 名
  ≠ 运行时 `_C` 的 **76** 名——**类型桩是不完整的面**。

> **H100 实测（E4）**：直接 import 真实的 `generate_pyi.py`、对 `root=csrc` 跑一遍，
> 产出的 `.pyi` 与已提交的 `stubs/_C.pyi` **逐字节相等**（diff 0 行）；复核 62 def / 14 fallback /
> pyi 无 `m.attr` / 源码 `m.def`=62、`m.attr` 赋值=10。见 `.out.txt` E4 段。

### 5.2 一处**不可复现**的工具 bug（E4 顺带发现）

但「逐字节相等」只在本机成立。**同一个脚本、同一份源码，换一台机器就可能产出不同的 pyi**——
因为 `build_cpp_function_index` 用 `Path(root).rglob('*')` 遍历文件，**遍历顺序依赖文件系统**，
而它对每个 base name **只保留第一次命中的签名**（`if base_name not in func_index`）。问题出在
`get_tma_aligned_size`：

- 它的**真定义**在 `csrc/utils/math.hpp:23`（`static int get_tma_aligned_size(const int& x, const int& element_size)`）；
- 但 `csrc/jit_kernels/impls/sm100_mqa_logits.hpp:240` 有一处**调用点**
  `make_tma_2d_desc(sf_kv, get_tma_aligned_size(num_kv_tokens, ...))`，正则的
  「语句边界」检查（`content[...].rstrip()` 末尾非字母数字）在这里**误判通过**，于是
  `get_tma_aligned_size` 的索引项被写成一段**乱码调用点**。

如果 `jit_kernels` 目录先于 `utils` 被遍历到，索引里就是乱码 → 该函数在 pyi 里退化成
`def get_tma_aligned_size(*args, **kwargs) -> Any: ...`；反之则是正确签名。**我们在两台环境上分别
观察到了这两种结果**（本机 host 命中调用点、实验容器命中真定义），并用「把索引里这一项强制替换成
调用点签名」**确定性地复现**了退化成 `*args` 的差异（且只差这一个函数）。

> `.out.txt` E4d 段：`build_cpp_function_index(csrc/utils)` → 真定义；
> `build_cpp_function_index(csrc/jit_kernels)` → 调用点；强制构造后重新生成，
> diff **恰好 8 行、唯一变化函数 = `get_tma_aligned_size`**。结论：**`stubs/_C.pyi` 不是可复现产物**，
> 依赖目录布局；修法是让索引只接受「定义」、或在同分时按路径优先级取定义。

### 5.3 `csrc/indexing/main.cu` + CMake（CUDA kernel 面）

`csrc/indexing/main.cu`（36 行）是一个**把所有 kernel impl include 进来的单 TU**：

```cpp
// GEMM kernels
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>
... (共 17 个 #include)
using namespace deep_gemm;
int main() { return 0; }
```

CMake 里（`CMakeLists.txt`）：

```cmake
# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
...
# Enable kernel code indexing with CMake-based IDEs
cuda_add_library(deep_gemm_indexing_cuda STATIC csrc/indexing/main.cu)
```

**它不参与发布**：真正的编译走 DeepJIT（第 23 篇）。它存在的唯一目的是给 IDE（CLion）一个**单一根**，
把所有 CUDA kernel 模板引进来做代码索引/跳转。它也是「这个项目一共编哪些 kernel」的**事实清单**。

> **H100 实测（E4f）**：`main.cu` 只 include 了 **17** 个 impl，而磁盘上 `impls/` 有 **19** 个
> `.cuh`——**漏了 `sm100_mega_mhc.cuh` 和 `sm100_sparse_mqa_logits.cuh`**（较晚加入的两个）。这说明
> 这份「kernel 面清单」也会**随代码演进腐化**（IDE 索引不报错，只是跳不到那两个文件）。见 `.out.txt` E4f 段。

---

## 6. 可跑性与边界

| 组件 | 架构相关性 | 本机 H100 结论 |
|---|---|---|
| `run_ncu_mega_moe.sh` 的**命令行构造** | 纯 bash | **真机验证**（桩捕获 argv，E1） |
| `run_ncu_mega_moe.sh` 的**目标 kernel** | SM100 + DeepEP | **仅分析**（端到端跑不了） |
| `test_mega_moe.py` 的 `--ncu-profile-only` 分支 | 需 SM100 | 源码解读；**性能账目公式**真机抽取执行（E3） |
| `quick_plot_pm.py` 的**解析/取址/画图** | 架构无关（除 chip 名） | **真机验证**（真实 `.ncu-rep` + raw CSV，E2） |
| `generate_pyi.py` / `stubs/_C.pyi` | 纯 Python | **真机逐字节复现** + 发现不可复现 bug（E4） |
| `csrc/indexing/main.cu` / CMake | 纯文本 | **真机核对** include 面（E4f） |

一句话：**这套 profiling 设施的「外壳」（脚本参数、报告解析、代码生成）全部在 H100 上真实验证过；
只有它要点的内核本身是 SM100 专属，端到端仅分析。**

---

## 7. 与其它篇的关系

- 第 3/9/10/11 篇：Mega MoE 主 kernel 的**内部**（warp、mbarrier、MMA、dispatch/combine）。
  本篇是**外部**：怎么观测它。
- 第 15 篇：反向 SM90 kernel 的 `ncu` 归因（`--section SpeedOfLight` + 定向 metric 的**平均值**）。
  本篇是**同一个 ncu 的另一半**——`PmSampling`（时间序列）+ `SourceCounters`（源码归因）+ 多进程编排。
- 第 23 篇：DeepJIT。`DG_JIT_WITH_LINEINFO=1` 就是 DeepJIT 的一个编译开关。
- 第 30/40 篇：`bench_kineto`。`--dump-profile-traces` 走的就是它。
- 第 41 篇：`test_mega_moe.py` 的**正确性**判据。本篇是**同一个测试文件**的**性能/工具**面。
- 第 84 篇：`_C` Python API 面的**人工枚举（76 名）**。本篇 5.1 用 `generate_pyi.py` 给出**机器生成**
  的 api 面（62 名），并指出它**漏了 10 个别名**、且**不可复现**——两者互补。
