# DeepGEMM 的 per-warp `%clock64` 追踪探针：`common/probe.cuh`

> 本文对应路线图 **P45-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 55 篇。
> 前面 0~54 篇把 GEMM / attention / grouped 反向 / Mega MoE / Mega Gate / mHC 的 kernel 与
> host/调度器/测试基础设施都拆过，期间**多次依赖 `%clock64` 做时序与标定**——第 24 篇的局部域
> pointer-chase 探针、第 26 篇 `set_tc_util` 的 tensor-core 节流空转、第 34 篇的域延迟测量、
> 第 52 篇的 PDL 时序实验。而 DeepGEMM 自己其实提供了一套**通用的 per-warp trace 设施**
> `deep_gemm/include/deep_gemm/common/probe.cuh`：把 `%clock64` 写成一串定长记录到全局内存，
> 两端用 `mark()`（`%clock64` + `%globaltimer`）做时钟标定。
>
> 奇怪的是：**公开仓库里这个文件没有任何调用点，JIT 也从不定义它依赖的 `DG_PROBE_ENABLE`**
> ——它是一段**休眠的观测基础设施**。本文把它逐行拆透，并在 H100(sm90) 上**直接编译运行真实
> 设备代码**、逐条验证语义与开销。
>
> **可跑性**：全部是架构无关的设备内联 asm（`mov %clock64` / `mov %globaltimer` / `st.global` /
> `add.u32`），sm90 可编译可运行；本机实测 **ALL PASS**。（`common/probe.cuh` 文件名里没有
> `sm100`，本身也不依赖 UMMA/TMEM，只是被 SM100-only 的 Mega MoE 用到而随版本发布。）

---

## 1. 为什么单开一篇：一段「编译为空」的观测设施

前面各篇的 profiling 大多靠 `torch.profiler` / `ncu` / `bench_kineto`（第 15/40 篇）。但有些
问题必须在**设备内核内部**测量：某个 warp 在某段代码停留多久、某个 task 在 ring buffer 里排队
多久、跨 SM 的 readiness 何时翻转。这类问题只能靠 `%clock64` 打点。DeepGEMM 的 `probe.cuh`
就是为此准备的**统一设施**，设计目标非常明确（源码注释 `:8`）：

> *Probes compile to nothing unless the JIT defines `DG_PROBE_ENABLE 1`.*

也就是说，它要满足：**生产构建零开销、零代码、零缓存键变化**（默认 `DG_PROBE_ENABLE` 未定义时
`kEnabled=false`，所有方法体走 `if constexpr (kEnabled)` 分支被整段消除），只有需要时用 JIT flag
打开。这与第 26 篇 `set_tc_util`（进模板参数、改 kernel）不同——probe 是**纯编译期开关 + 独立
缓冲区**，甚至不进 kernel 的语义。

本篇要讲清三件事：① 记录流的内存格式与「先 advance 再负偏移回填」的约定；② `is_valid_buffer`
的三条安全约束**尤其「必须落在同一个 4 GiB 窗口内」**从何而来；③ `mark()` 如何用
`%globaltimer`（纳秒挂钟）标定 `%clock64`（SM 周期）。

---

## 2. 文件精读（`common/probe.cuh`，91 行）

### 2.1 编译开关（`:8-11`）

```cpp
// probe.cuh:8-11
// Probes compile to nothing unless the JIT defines `DG_PROBE_ENABLE 1`
#ifndef DG_PROBE_ENABLE
#define DG_PROBE_ENABLE 0
#endif
```

`kEnabled`（`:19`）= `DG_PROBE_ENABLE and kNumRecords > 0`——**容量为 0 的 Stream 也恒关**，
这样「用一个 `Stream<0>` 占位」不会引入任何代码。

### 2.2 三个静态量与 `get_num_bytes`（`:17-25`）

```cpp
// probe.cuh:17-25
template <uint32_t kNumRecords>
struct Stream {
    static constexpr bool kEnabled = DG_PROBE_ENABLE and kNumRecords > 0;
    static constexpr uint32_t kNumBytesPerRecord = 8;        // 每条记录 8 字节
    static constexpr uint32_t kNumBytes = kNumRecords * kNumBytesPerRecord;
    CUTLASS_HOST_DEVICE static constexpr uint64_t get_num_bytes(const uint32_t& num_streams) {
        return static_cast<uint64_t>(num_streams) * kNumBytes;   // 多流 = 多 warp 各一段
    }
```

一条记录就是**一个 64 位字**（记 `%clock64`）。多流时总大小 = `num_streams × kNumBytes`
（本机实测 `<64>`：`num_bytes(1)=512`、`num_bytes(3)=1536`，逐点吻合）。缓冲区由**调用方**
分配并清零（注释 `:16`），probe 只负责写。

### 2.3 `is_valid_buffer`：三条安全约束（`:27-31`）

这是全文件最值得讲的一处：

```cpp
// probe.cuh:27-31
CUTLASS_HOST_DEVICE static bool is_valid_buffer(const void* buffer, const uint32_t& num_streams) {
    const auto base = reinterpret_cast<uint64_t>(buffer);
    return kNumRecords % 2 == 0 and base % 16 == 0 and
           (base >> 32) == ((base + get_num_bytes(num_streams) - 1) >> 32);
}
```

三条约束各有来源：

| 约束 | 来源 |
| --- | --- |
| `kNumRecords % 2 == 0` | `record_payload` 以 16 字节写入（两条记录），容量须为偶数整块 |
| `base % 16 == 0` | 同上，`st.global.v4.b32` 要求 16 字节对齐 |
| `(base>>32) == (end>>32)` | **`advance` 只加低 32 位**（见 §2.6），整段必须落在同一个 4 GiB 窗口 |

第三条是**先有 `advance` 的实现、后有这个约束**——见 §3.1。本机用真实指针 + 伪指针逐条验证了
这三条（§5 D）。注意 `record_payload` 的 compile-time 断言 `DG_STATIC_ASSERT(kRecordOffset%2==0)`
（`:56`）只保证**偏移是偶数条记录**，真正要求 16 字节对齐的是「`cursor` 本身也落在偶数槽」——
所以你在 §2.5 的 trace 里会看到：调用 `record_payload` 之前必须把 cursor 推进到偶数槽。

### 2.4 构造函数与「关掉就整段消失」（`:33-38`）

```cpp
// probe.cuh:33-38
uint64_t cursor = 0;
CUTLASS_DEVICE Stream(void* buffer, const uint32_t& stream_idx) {
    if constexpr (kEnabled)
        cursor = reinterpret_cast<uint64_t>(buffer) + static_cast<uint64_t>(stream_idx) * kNumBytes;
}
```

关闭时 `cursor` 保持 0，且**后续所有方法都是 no-op**——本机实测：`DG_PROBE_ENABLE=0` 时
`advance` 后 `cursor` 仍是 0、trace 缓冲区**全零**（§5 B2/E）。

### 2.5 `record`：先 advance、再用负偏移回填（`:40-50`）

```cpp
// probe.cuh:40-50
// NOTES: advance the cursor first and record with negative offsets, so that ptxas keeps the add in place
template <int32_t kRecordOffset = 0>
CUTLASS_DEVICE void record() const {
    if constexpr (kEnabled) {
        asm volatile("{\n\t"
                     ".reg .b64 t;\n\t"
                     "mov.u64 t, %%clock64;\n\t"
                     "st.global.b64 [%0+%1], t;\n\t"
                     "}" :: "l"(cursor), "n"(kRecordOffset * kNumBytesPerRecord) : "memory");
    }
}
```

- `record<off>` 写 `cursor + off*8`，**不改 cursor**（改 cursor 是 `advance` 的事）。`off` 由 `"n"`
  约束编成**立即数**，所以是编译期常量偏移。
- 注释里的约定是：「**先 `advance<N>()` 跳过 N 个槽，再用 `record<-N..-1>()` 回填**」。这样
  `%clock64` 的读取与 `st.global` 的地址计算被拆开，`advance` 的加法能在 ptxas 里**原地保留**
  （因为后续记录用固定的负偏移，地址可复用）。
- `record<0>()` 后必须手动 `advance<1>()` 才能推进到下一个槽——本机 trace 里两种写法都用了
  （§5 B）。

### 2.6 `record_payload`（`:52-61`）与 `mark`（`:63-75`）

```cpp
// probe.cuh:52-61
// Two records of payload (e.g. a task descriptor) for attributing the surrounding records
template <int32_t kRecordOffset>
CUTLASS_DEVICE void record_payload(const uint32_t& a, const uint32_t& b,
                                   const uint32_t& c, const uint32_t& d) const {
    DG_STATIC_ASSERT(kRecordOffset % 2 == 0, "Payloads are 16-byte aligned");
    if constexpr (kEnabled) {
        asm volatile("st.global.v4.b32 [%0+%1], {%2, %3, %4, %5};"
                     :: "l"(cursor), "n"(kRecordOffset * kNumBytesPerRecord), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
    }
}
```

`record_payload` 一次写 16 字节（`v4.b32`），用来把**周围的 clock 记录**和某个 payload
（注释举例：task 描述符）关联起来——这样事后解析时能知道「这段时间在算什么任务」。

```cpp
// probe.cuh:63-75
// `{clock64, globaltimer}` for calibrating the SM clock, as two 8-byte stores (the cursor may not be 16-byte aligned)
CUTLASS_DEVICE void mark() {
    if constexpr (kEnabled) {
        asm volatile("{\n\t"
                     ".reg .b64 t, g;\n\t"
                     "mov.u64 t, %%clock64;\n\t"
                     "mov.u64 g, %%globaltimer;\n\t"
                     "st.global.b64 [%0], t;\n\t"
                     "st.global.b64 [%0+8], g;\n\t"
                     "}" :: "l"(cursor) : "memory");
        advance<2>();
    }
}
```

`mark()` 写成**两条 8 字节 store**（不是一条 `v4.b32`），因为注释 `:63` 说 cursor 可能不是
16 字节对齐——两条独立 8 字节对齐的 store 更安全。它一次占两个槽（`clock64` + `globaltimer`），
所以末尾 `advance<2>()`。**`%globaltimer` 是纳秒级挂钟**，这正是标定 `%clock64` 频率的锚点：

> `频率(GHz) = Δ%clock64 / Δ%globaltimer(ns)`

本机实测两组独立标定都得到 **1.98 GHz**（§5 C）。

### 2.7 `advance`：只加低 32 位（`:77-88`）

```cpp
// probe.cuh:77-88
// Only the low 32 bits are advanced (a single `IADD3`)
template <uint32_t kNumRecordsToAdvance>
CUTLASS_DEVICE void advance() {
    if constexpr (kEnabled) {
        asm volatile("{\n\t"
                     ".reg .b32 lo, hi;\n\t"
                     "mov.b64 {lo, hi}, %0;\n\t"
                     "add.u32 lo, lo, %1;\n\t"
                     "mov.b64 %0, {lo, hi};\n\t"
                     "}" : "+l"(cursor) : "n"(kNumRecordsToAdvance * kNumBytesPerRecord));
    }
}
```

**这是全文件的实现核心**：把 64 位 cursor 拆成 `lo/hi`，只对 `lo` 做 `add.u32`，再拼回去。
一条 `IADD3` 就完成——**高 32 位永不进位**，所以只要缓冲区不跨 4 GiB 边界，低 32 位加法就
等价于完整 64 位加法。这就是 §2.3 第三条约束的根因。

本机用**伪指针**直接实证（`advance` 不解引用，可以传任意 64 位值当 cursor，§5 E）：

| 起始 cursor | `advance<2>()`（+16）后 | 说明 |
| --- | --- | --- |
| `0x00000001_FFFFFFF0` | `0x00000001_00000000` | 低 32 位回绕，**高 32 位不动**（偏移「倒退」4 GiB） |
| `0x00000000_FFFFFFF0` | `0x00000000_00000000` | 低 32 位回绕到 0 |
| `0x0000FFF3_00000000` | `0x0000FFF3_00000008` | 正常 +8 |

第一行就是「为什么必须有 4 GiB 窗口约束」的直接证据。

---

## 3. 把「为什么」串起来

### 3.1 4 GiB 窗口 ← 单条 `IADD3` 的代价

`advance` 想要「一条加法指令」，代价就是**放弃高 32 位进位**。于是安全性必须由
`is_valid_buffer` 的 `(base>>32)==(end>>32)` 来保证。这是典型的「用可验证的调用契约换内核里
一条指令」——和目标卡上很多「TMA/描述符对齐契约」（第 12/25 篇）是同一设计哲学。

### 3.2 负偏移回填 ← 让 ptxas 保留那条加法

`record` 本身不改 cursor，改 cursor 只在 `advance`。若每次 `record` 都自带一次 `advance`，则
`cursor` 更新依赖上一次记录，形成长依赖链；而「先一次性 `advance<N>`、再以固定负偏移回填」
让地址 = `cursor_基址 + i*8` 全在同一基址上，ptxas 只需算一次基址。对 per-warp 高频打点，这
能显著压低开销（§5 F 实测纯记录 ~5.7 ns/条）。

### 3.3 `mark` 用挂钟标定 ← 墙时间与 SM 周期解耦

`%clock64` 是 SM 周期计数，频率随 boost/负载变化；`%globaltimer` 是纳秒挂钟。两端各打一个
`mark`，就能把一段 trace 的「周期数」换算成「真实时间」，或在比较跨 SM/跨 launch 的 trace 时
统一到同一时基。第 24 篇的局部域探针也是「`clock64` 除跳数得每跳周期」（`sm100_locality_domain.cuh`），
但那里没做挂钟标定；`probe.cuh` 的 `mark` 补上了这一环。

---

## 4. 与其它 `%clock64` 用法的关系

| 篇目 | 用处 | 与 probe 的关系 |
| --- | --- | --- |
| 第 24 篇 局部域探针 | pointer-chase 每跳周期 | 单独内联 `clock64`；probe 是其「通用化 + 标定」版 |
| 第 26 篇 `set_tc_util` | tensor-core 节流空转计时 | 节流用 `clock64` 空转；probe 可测其真实占空比 |
| 第 52 篇 PDL | 同一 kernel 两段落时序 | 可用 probe 记录「griddepcontrol.wait 前后」 |
| **本篇** | **通用 per-warp trace 设施（休眠）** | **唯一提供 `mark` 挂钟标定 + 定长记录流** |

注意 `common/probe.cuh` 与 `impls/sm100_locality_domain.cuh` 的探针**是两套东西**：后者是
「延迟测量算法」（chase 指针链），前者是「通用计时/打点基础设施」。

---

## 5. H100 实测（`src/probe_clock64.cu` + `harness/exp_probe_clock64.sh`）

在 kernel_lab（H100 80GB HBM3, sm90, driver 570.211.01）上以 `nvcc -arch=sm_90 -O3` 编译，
**同一份源码分别以 `-DDG_PROBE_ENABLE=1` 与默认（0）编成两个二进制**，Python 侧用 numpy 解析
binary 记录流。`probe.cuh` 的 sha256 = `a8c3ba24…`。**总判定 ALL PASS**：

| 组 | 内容 | 实测 |
| --- | --- | --- |
| A | `kEnabled<8>=1` / `kEnabled<0>=0`；`kNumBytesPerRecord=8`、`kNumBytes<8>=64` | 全对 |
| B | trace 记录流：payload 四字 `0x11111111/22222222/33333333/44444444` 精确还原、clock 单调、负偏移回填 `r4≥r3 & r5≥r4`、globaltimer 单调 | **True×4** |
| B2 | `DG_PROBE_ENABLE=0` 时 trace 缓冲区**全零** | **True** |
| C | `mark()` 挂钟标定：`cycles=20000044, ns=10101088` → **1.9800 GHz**；trace 两端 marks → **1.9796 GHz** | 均在 1~3 GHz |
| D | `is_valid_buffer` 六例 + `get_num_bytes` | 逐条 = 期望 |
| E | `advance` 伪指针：`[0x0000000100000000] [0x0] [0x0000fff300000008]` | **精确命中**；关闭时 cursor 恒 0 |
| F | 开销：fma 循环 on 63.52 / off 56.79 ms；纯记录 4 warp×4096：on 0.095 / off 0.0017 ms | 见下 |

**D 组逐条**（`is_valid_buffer`）：

```
real_aligned      = 1      # 真实 cudaMalloc 指针，256 对齐
real_misaligned8  = 0      # base%16 != 0
odd_records       = 0      # Stream<63>（奇数容量）
crosses_4gib      = 0      # base=0xFFFFFFF0，64 条(512B) 越过 4 GiB 边界
near_but_inside   = 1      # base=0xFFFFFFF0-512，仍落在同一窗口
inside_window     = 1      # 低地址
multi_stream_stay = 1      # 3 条流 1536B 仍在窗口内
num_bytes_1 = 512, num_bytes_3 = 1536
```

**B 组记录流**（64 槽，仅列关键槽）：

```
r0 = clock(mark0)   r2/r3 = record<0> 两次   r4/r5 = advance<2> 后负偏移回填
r6,r7 = payload 两条 8B 字     r8 = nanosleep(20us) 后 record
r9,r10 = clock/globaltimer(mark1)
clock 序列(r0,r2,r3,r4,r5,r8,r9) 严格单调；globaltimer mark0→mark1 Δ=31840 ns
```

**F 组**：两处都证明「开 probe 只在被记录处多一条 `st.global.b64` + 一条 `IADD3`」——
- `probe_only`（纯记录，4 warp 各 4096 条）：开 **0.0947 ms** vs 关 **0.0017 ms**（关的就是
  一次 launch 的噪声）；差 0.093 ms / 16384 条 ≈ **5.7 ns/条**（含 4 warp 并发的墙时间）。
- fma 循环版：单线程 + 单 warp，绝对时间被 **idle clock（~0.35 GHz）** 主导（56.79 ms），
  所以「+33.6 ns/条」偏大，但**方向正确**：探针开销 ~ns 级，且当循环本身够长时可被完全隐藏。

> 一个细节：`record<-2>`/`record<-1>` 时 nvcc 会对 `"n"(负数)` 报
> `warning #68-D: integer conversion resulted in a change of sign`（无害，只是负立即数）。
> 源码用它作编译期偏移，运行时完全正确（B 组 `r4/r5` 回填正确）。

---

## 6. 边界与「仅分析」

- **全文件架构无关**：`mov %clock64` / `mov %globaltimer` / `st.global` / `add.u32` 在 sm90 上
  全部合法，本机可编译可运行（本篇即真机实测）。它之所以随 SM100-only 的 Mega MoE 发布，
  只是因为**调用点在 SM100 kernel 里**，不是因为它依赖 SM100。
- **公开仓库无调用点**：`grep -r "common/probe.cuh\|DG_PROBE_ENABLE"` 全仓库只有 `probe.cuh`
  自身与 JIT 里**未设置**的 flag（`csrc/runtime/jit.hpp:13-33` 只加了两条 `--diag-suppress` /
  `-Wno-deprecated-declarations`）。所以本项目**没有实测其真实调用形态**，只验证了设施本身的
  语义与开销；若要接入，需要 JIT 侧加 `-DDG_PROBE_ENABLE=1` 并在 kernel 里显式建 `Stream`。
- **4 GiB 窗口**是**契约**而非运行时检查：`is_valid_buffer` 只是给调用方的**断言辅助**，
  device 侧 `advance` 不做任何检查（越界会静默回绕）。
- **未覆盖**：多流（`stream_idx>0`）只在 host 侧验证了 `get_num_bytes`；device 侧 `probe_only`
  用了 4 warp 各写各的流但未逐条核对流间隔离（`cursor = base + idx*kNumBytes` 的偏移逻辑简单，
  已由 host 公式坐实）。

---

## 7. 参考

- DeepGEMM 仓库：`https://github.com/deepseek-ai/DeepGEMM`（本地 commit `057ca59`，
  Public release 26/09/30，v2.1.1.post3-24）。
- 源码：`deep_gemm/include/deep_gemm/common/probe.cuh`（91 行，sha256 `a8c3ba24…`）；
  JIT 接入点 `csrc/runtime/jit.hpp:13-33`（未定义 `DG_PROBE_ENABLE`）。
- 实验：`code/mega-moe/src/probe_clock64.cu`、`code/mega-moe/harness/exp_probe_clock64.{sh,out.txt}`。
- 相关篇目：第 15/40 篇（profiling 基座）、第 24 篇（局部域 pointer-chase 探针）、
  第 26 篇（`set_tc_util` 节流空转）、第 34 篇（域延迟测量）、第 52 篇（PDL 时序）。
