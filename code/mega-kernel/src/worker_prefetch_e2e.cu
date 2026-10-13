// worker_prefetch_e2e.cu — MegaKernel 专题 P66-1 小实验：
//   补掉 §54.8（P65-1）明留的尾巴「worker 取 task 环深固定 `D=1`（见坑②）：**预取深度是 §41/§42 的
//   独立变量**；本节为端到端**同步对照**固定 D=1，故绝对吞吐含一个每批的串行缝（§43 量过，重 task 下
//   可藏、轻 task 下显形）」，以及坑②「worker 取 task 用**跨步分配** `bb = w + t*Gw`，`D>1` 预取时
//   worker 会在 `issue(t+D)` 处**先于消费任何一批**就等一个跨出 producer 子环边界的 `head` →
//   **有界环死锁**」。
//   即：把 §54 端到端流水里那个被钉死为 `D=1` 的「worker 取 task」维度**放开**，量两件事：
//     (1) **预取深度 `D`**：加深取 task 环能否把 §43「每批串行缝」藏住、抬高端到端吞吐（worker 侧）；
//     (2) **分配形态 `wassign`**：跨步（strided，`bb=w+t*Gw`）vs 连续分块（contig，`bb=w*chunk+t`）
//        在有界环 + 动态生产者下谁是安全形态（坑②的死锁边界随分配变化）。
//
// 背景（§54 结论）：有界环绑定时端到端由「背压×容量乒乓」主导（~20×）、与 P 几乎无关；环容量不绑定
//   时复现 §53 内存序归因但衰减到 ~1.1–1.2×。**但全部固定 D=1**——本实验放开 D。
//
// ----------------------------------------------------------------------------
// 以下原始内容来自 dyn_producer_carrier_e2e.cu（P65-1），保留 mode 9 纯写、mode 21 端到端与全部参照：
//
// dyn_producer_carrier_e2e.cu — MegaKernel 专题 P65-1 小实验：
//   把 §52/§53（P63-1/P64-1）钉死的**写端握手归因**从**纯写口径（mode 9，grid=P，无 worker）**
//   推进到**端到端（有 worker + 数据面）**，补掉 §53.9 明留的尾巴「纯写口径……带 worker 的端到端
//   归因未在本节跑（§52.8 另一条尾巴，留待后续）」，并回答：
//   动态生产者写端在临界路径插入的握手税（每批 release / 背压会合），**在有 worker 与数据面时还看得见吗**
//   —— 是仍决定端到端，还是像 §44–§46「控制面被数据面藏住」那样消失？
//
// 背景（§52/§53 纯写结论，P=128）：
//   - store-only 地板 `nobp` = 630 GB/s（TMA）；正确 serialized `ser` = 437；
//   - 三载体（TMA/寄存器 copy-out/寄存器直写）分解**同形**：`ser < relhead < skipbp≈skiphd < nobp`，
//     `ld.acquire`→relaxed 免费、release 才是主因；握手税近似**每批固定绝对量**（~2.0–2.4k cycles）。
//   - **但这些都是 pure-write（无 worker）**：§53.9 明确「带 worker 的端到端归因未跑」。
//
// 本节：同一 §52 骨架 + 六位正交编码（WA/BP/HD/REL/ACQ/FENCE），新增 **mode 21 = 动态生产者
//   (gen) + G worker + 可选数据面**，把 §53 的归因矩阵在**端到端**重跑：
//   - worker 用 TMA bulk load 取回描述符并执行（控制面折叠 + 数据面 `uint4` 流式读 + R 轮压舱）；
//   - 端到端墙钟 = `max(producer 写侧, worker 取+执行侧)`；扫数据面 EDW 与 producer 数 P，
//     看写侧握手税(ser vs nobp_wa/skipbp)是否随 worker/数据面变重而被隐藏（临界路径归属）。
//   - 哨兵档（故意不可靠、只拆代价）**必须保留发布(HD)**，否则 worker 等不到 head 会挂起。
//
// 安全：mode 21 producer 有 clock64 无进度守门狗，脚本带 timeout。
// 编译：nvcc -O3 -arch=sm_90a src/dyn_producer_carrier_e2e.cu -o /tmp/dyn_producer_carrier_e2e
//
// ----------------------------------------------------------------------------
// 以下原始内容来自 dyn_producer_carrier_dep.cu（P64-1），保留 mode 9 纯写路径与 cbench 作参照：
//
// dyn_producer_carrier_dep.cu — MegaKernel 专题 P64-1 小实验：
//   把 §52（P63-1）钉死的**写端握手归因**从**单一载体（TMA bulk store）**推进到**三种写载体**。
//   同一 §52 骨架 + 六位正交编码，新增 **`carrier`** 维度（同一 D=2、同一发布协议）：
//   - carrier 0 = **TMA bulk store**（SMEM 暂存 → `cp.async.bulk` 引擎，= §52）
//   - carrier 1 = **寄存器 copy-out**（SMEM 暂存 → 全 block `uint4` load/store）
//   - carrier 2 = **寄存器直写**（不落 SMEM；每线程生成 `uint4` 直接写环 = §49/§50 v4 形态）
// 编译：nvcc -O3 -arch=sm_90a ... -o /tmp/dyn_producer_carrier_dep
//
// ----------------------------------------------------------------------------
// dyn_producer_tma_dep.cu — MegaKernel 专题 P63-1 小实验：
//   把 §51（P62-1）留下的**开放问题**钉到「内存序 / 依赖链」一级。
//
// 接续（补掉 §51.8 明留的开放问题）：
//   §51.6（P62-1）在 TMA store 动态生产者的纯写口径（P=128）上量出一组反直觉的归因：
//     - serialized 正确握手 ser = **453.6 GB/s**；
//     - **只拆背压**（skipbp = 去掉 tail 的 `ld.acquire` 轮询 + 该批的 `__syncthreads()`）→ **664.5**；
//     - **只拆 head 发布**（skiphd = 去掉 `bulk_wait_all` + `fence.proxy.async.global` + `st.release`）
//       → **664.8**；
//     - **两者都拆**（nobp）→ 664.3。
//   即「拆掉任**一个**就够快」，而两者**同时存在**才昂贵。§51.6 用 ncu 给出**症状**（barrier 停顿
//   31.19% vs 11.85%、多出 `membar`/`long_scoreboard`），但自陈「未到 SASS/依赖链一级给出**因果**，
//   可能是一个 loop-carried 依赖阻止了编译器软件流水。**记为开放问题**」。本节把它钉死。
//
// 问题：到底是「协议簿记」的**哪一维**？本节约四个正交假设：
//   H1 **跨 SM 内存序语义**：贵的是 `st.release head` / `ld.acquire tail` 这对 release/acquire 的
//       **ordering 语义本身**（换 `st.relaxed`/`ld.relaxed` 即可恢复）。
//   H2 **每批一次会合（barrier）**：贵的是背压那一步的 `__syncthreads()`（把背压按 S 批摊薄即可恢复）。
//   H3 **每批一次对外报进度**：贵的是**每批**发布 head（把发布按 S 批摊薄即可恢复）。
//   H4 **两个 SASS 指令的存在**：贵的是那两个 strong-memory 指令本身的发射（SASS 计数）。
//
// 设计（载体固定 TMA bulk store = gvar 3；其余与 §47–§51 同骨架）：
//   一组**哨兵口径**（故意不保证完成可见性，只做**代价归因**，与 §51 的 `win<0` 同性质）：
//     - `win=1`  ser：正确 serialized（每批 wait_all + fence + release head）
//     - `win=-4` skipbp：去掉背压（tail acquire + 该批 syncthreads），保留 head 发布
//     - `win=-5` skiphd：去掉 head 发布（含 wait_all/fence），保留背压
//     - `win=-6` relhead：保留背压与发布**动作**，但 head 用 `st.relaxed.gpu`（非 release）→ 测 H1
//     - `win=-7` reltail：tail 用 `ld.relaxed.gpu`（非 acquire），head 仍 release → 测 H1
//     - `win=-8` relboth：两者都 relaxed → 测 H1
//     - `win=-9` stridedbp：背压每 `S` 批做一次（`S=bpstride`），head 仍每批 → 测 H2
//     - `win=-10` stridedpub：head 每 `S` 批发布一次（发布累积到 `(bj+1)*K`），背压仍每批 → 测 H3
//     - `win=-11` stridedall：背压与发布都摊薄到每 `S` 批 → H2+H3 合取
//   另加 **`mode 11/13` = `stridedbp`/`stridedpub` 的**“正确”版本**（预留 `S` 批容量 + 累积发布），
//   在**有 worker** 的口径下跑 §A 正确性（`got` 与 static 逐位一致）。
//
//   逐相位计时（`instr=1`，lane0 用 `clock64()`）把每批拆成：`bp`（背压）/`wr`（回收槽 wait_read）/
//   `gen`（协作生成 SMEM）/`store`（bulk store 发射 + commit）/`pub`（wait_all + fence + release）/
//   `pw`（压舱计算）。用于看 H2/H3 各自落在哪一段。
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：MPK 队列协议本身就是
//   「`st.relaxed.gpu` 写数据 + 单条 `atom.add/.cas.release.gpu` 推游标」，靠**批量/环形**摊薄一次
//   release/acquire（`docs/06` §5.3、本专题 P47-1 §36）。本节把「摊薄」这件事逐相位拆开：
//   若结论是 H1（release/acquire 语义），则「批量摊薄」仍有效；若是 H2（会合），则要改**通信结构**
//   （单写者不共享 SMEM 等），而非只调 batch 大小。
//
// 安全：大 CAP 关掉背压只量 producer 吞吐；producer 有 `clock64` 无进度守门狗防死锁挂死，脚本再带 `timeout`。
//
// 编译：nvcc -O3 -arch=sm_90a src/dyn_producer_tma_dep.cu -o /tmp/dyn_producer_tma_dep
// 用法：./dyn_producer_tma_dep run <P> <G> <NDESC> <DW> <EDW> <K> <D> <R> <PW> <CAP> <mode> <reps>
//          [T] [gvar] [df] [bpstride] [win] [sb] [instr]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t _e = (x);                                                             \
    if (_e != cudaSuccess) {                                                          \
      printf("CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));   \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

#define MAXD 64
#define MAXP 132
#define SB 4  // TMA store 的 SMEM 暂存槽数（环形，最多 SB-1 个 store 在飞）

__host__ __device__ __forceinline__ uint32_t mix(uint32_t s, uint32_t i) {
  uint32_t x = s * 2654435761u + i * 40503u + 0x9e3779b9u;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// 第 gid 条描述符的第 i 个 32-bit 字（所有写入形态共用，保证 payload 内容一致）。
__host__ __device__ __forceinline__ uint32_t gen_word(uint32_t gid, int i) {
  if (i == 0) return gid;
  if (i == 1) return gid % 4u;
  if (i == 2) return gid / 4u;
  if (i == 3) return 0u;
  return mix(gid, (uint32_t)i);
}

// ---- 内存序原语（对齐 MPK 队列协议；快照 mpk_atoms.cuh:17-96）----
// acquire/release 档
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
// relaxed 档（本节哨兵：只变 ordering 语义，不变搬运量）
__device__ __forceinline__ unsigned long long ld_relaxed_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.relaxed.gpu.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long atom_add_release_u64(unsigned long long* p,
                                                                   unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.add.release.gpu.u64 %0, [%1], %2;" : "=l"(old) : "l"(p), "l"(v)
               : "memory");
  return old;
}

// ---- 硬件 mbarrier + TMA bulk load（worker 取 task）----
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(a), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(a),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "MBW_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra.uni MBD_%=;\n"
      "bra.uni MBW_%=;\n"
      "MBD_%=:\n"
      "}\n" ::"r"(a),
      "r"(phase));
}
__device__ __forceinline__ void bulk_load_1d(void* smem_ptr, const void* gmem_ptr,
                                             uint32_t bytes, uint64_t* bar) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  uint32_t m = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], "
      "%2, [%3];\n" ::"r"(s),
      "l"(gmem_ptr),
      "r"(bytes),
      "r"(m)
      : "memory");
}

// ---- TMA bulk store（producer 写描述符）：SMEM -> global ----
__device__ __forceinline__ void bulk_store_1d(void* gmem_ptr, const void* smem_ptr,
                                              uint32_t bytes) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(gmem_ptr),
               "r"(s), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read_n(int n) {  // 等 store 读完源，允许 n 个在飞
  switch (n) {
    case 0: asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); break;
    case 1: asm volatile("cp.async.bulk.wait_group.read 1;\n" ::: "memory"); break;
    case 2: asm volatile("cp.async.bulk.wait_group.read 2;\n" ::: "memory"); break;
    case 3: asm volatile("cp.async.bulk.wait_group.read 3;\n" ::: "memory"); break;
    case 4: asm volatile("cp.async.bulk.wait_group.read 4;\n" ::: "memory"); break;
    case 5: asm volatile("cp.async.bulk.wait_group.read 5;\n" ::: "memory"); break;
    case 6: asm volatile("cp.async.bulk.wait_group.read 6;\n" ::: "memory"); break;
    case 7: asm volatile("cp.async.bulk.wait_group.read 7;\n" ::: "memory"); break;
    case 8: asm volatile("cp.async.bulk.wait_group.read 8;\n" ::: "memory"); break;
    default: asm volatile("cp.async.bulk.wait_group.read 15;\n" ::: "memory"); break;
  }
}
__device__ __forceinline__ void bulk_wait_all() {  // 等全部 store 完成（写可见）
  asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}
__device__ __forceinline__ void fence_proxy_async_global() {
  asm volatile("fence.proxy.async.global;\n" ::: "memory");
}

// 现场写一条描述符（gvar 0 标量 / gvar 1 v4），ptr 指向该 desc 起点。
__device__ __forceinline__ void write_desc(uint32_t* ptr, uint32_t gid, int DW, int gvar) {
  if (gvar == 0 || (DW & 3)) {
    for (int i = 0; i < DW; ++i) ptr[i] = gen_word(gid, i);
    return;
  }
  uint4* p4 = reinterpret_cast<uint4*>(ptr);
  const int NWU = DW >> 2;
  for (int u = 0; u < NWU; ++u) {
    uint32_t w0 = gen_word(gid, 4 * u), w1 = gen_word(gid, 4 * u + 1);
    uint32_t w2 = gen_word(gid, 4 * u + 2), w3 = gen_word(gid, 4 * u + 3);
    p4[u] = make_uint4(w0, w1, w2, w3);
  }
}

// 执行一个 batch：控制面折叠 + 数据面流式读 + R 轮 LCG 压舱。
__device__ __forceinline__ unsigned long long exec_batch(const uint32_t* desc, int K, int DW,
                                                         const uint32_t* dbase, int nlanes,
                                                         int lane, int EDW, int R, int bid,
                                                         int df) {
  uint32_t a = 0x9e3779b9u;
  if (df)
    for (int j = lane; j < K; j += nlanes)
      for (int i = 0; i < DW; ++i) a = a * 2246822519u + desc[(size_t)j * DW + i];
  uint32_t s = 0;
  const size_t totv = ((size_t)K * (size_t)EDW) >> 2;
  const uint4* dv = reinterpret_cast<const uint4*>(dbase);
  for (size_t i = lane; i < totv; i += nlanes) {
    uint4 x = dv[i];
    s += x.x + x.y + x.z + x.w;
  }
  uint32_t v = a * 2654435761u + s + (uint32_t)bid;
  for (int r = 0; r < R; ++r) v = v * 1664525u + 1013904223u;
  return (unsigned long long)a + (unsigned long long)v;
}

// variant 编码（producer gvar3 路径的发布/背压握手；WA=完成等待 BP=背压 HD=发布 REL/ACQ=内存序）：
//   win=1  ：ser           WA1 BP1 HD1 REL1 ACQ1（正确 serialized，= §51 的 ser）
//   win=-1 ：nowait        WA0 BP1 HD1          （= §51 nowait）
//   win=-2 ：nowa_relhead  WA0 BP1 HD1 REL0
//   win=-3 ：nowa_reltail  WA0 BP1 HD1 ACQ0
//   win=-4 ：skipbp        WA0 BP0 HD1          （= §51 skipbp）
//   win=-5 ：skiphd        WA0 BP1 HD0          （= §51 skiphd）
//   win=-6 ：nobp          WA0 BP0 HD0          （= §51 nobp）
//   win=-7 ：relhead       WA1 BP1 HD1 REL0
//   win=-8 ：reltail       WA1 BP1 HD1 ACQ0
//   win=-9 ：relboth       WA1 BP1 HD1 REL0 ACQ0
//   win=-10：stridedbp     WA1 BP1/S HD1
//   win=-11：stridedpub    WA1 BP1 HD1/S
//   win=-12：stridedall    WA1 BP1/S HD1/S
//   win=-13：nowa_strpub   WA0 BP1 HD1/S
//   win=-14：nowa_strbp    WA0 BP1/S HD1
// mode 11 = stridedbp 的**正确**版（有 worker 抽验）、mode 13 = stridedpub 的**正确**版。
__global__ void dptd_kernel(int P, int NDESC, int DW, int EDW, int K, int D, int R, int PW,
                            int CAP, int mode, int bp, int gvar, int df, int bpstride, int win,
                            int sb, int instr, int carrier, int wassign,
                            const uint32_t* __restrict__ static_ring,
                            uint32_t* __restrict__ ring, unsigned long long* __restrict__ head,
                            unsigned long long* __restrict__ tail,
                            const uint32_t* __restrict__ data,
                            unsigned long long* __restrict__ sink,
                            unsigned long long* __restrict__ psink, int* __restrict__ abort,
                            unsigned long long* __restrict__ ph) {
  extern __shared__ __align__(16) uint32_t smem[];
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int nb = NDESC / K;
  const int CAPp = (mode == 0) ? 0 : (CAP / P);  // 每 producer 子环槽数（K 的倍数）
  const bool is_gen = (mode == 1 || mode == 5 || mode == 8 || mode == 9 || mode == 11 ||
                       mode == 12 || mode == 13 || mode == 21);
  const bool smem_only = (mode == 6);
  // ---- 归因矩阵：显式位（win=1 正确 serialized；win<0 的哨兵档按下面映射）----
  //   WA 完成等待(bulk_wait_all)  BP 背压(tail acquire + syncthreads)  HD 发布 head
  //   REL head 用 release(1)/relaxed(0)  ACQ tail 用 acquire(1)/relaxed(0)
  bool WA = true, BP = true, HD = true, REL = true, ACQ = true, FENCE = true;
  int bps = 1, pbs = 1;  // 背压 / 发布的摊薄步长
  const int S = (bpstride < 1) ? 1 : bpstride;
  if (mode == 9 || mode == 21) {
    switch (win) {
      case -1: WA = false; break;                         // nowait（= §51 nowait）
      case -2: WA = false; REL = false; break;            // nowa + relhead
      case -3: WA = false; ACQ = false; break;            // nowa + reltail
      case -4: WA = false; BP = false; break;             // skipbp（真：无背压、留发布）
      case -5: WA = false; HD = false; break;             // skiphd（真：留背压、无发布）
      case -6: WA = false; BP = false; HD = false; break; // nobp
      case -7: REL = false; break;                        // relhead（WA 保持）
      case -8: ACQ = false; break;                        // reltail（WA 保持）
      case -9: REL = false; ACQ = false; break;           // relboth（WA 保持）
      case -10: bps = S; break;                           // stridedbp（WA 保持）
      case -11: pbs = S; break;                           // stridedpub（WA 保持）
      case -12: bps = S; pbs = S; break;                  // stridedall（WA 保持）
      case -13: WA = false; pbs = S; break;               // nowa stridedpub
      case -14: WA = false; bps = S; break;               // nowa stridedbp
      case -15: WA = false; BP = false; FENCE = false; break;  // 只剩 head release（无 fence）
      case -16: WA = true; BP = false; break;             // 无背压但完成等待 + 发布
      case -17: WA = true; BP = false; HD = false; break; // 无背压、无发布、只等完成
      default: WA = BP = HD = REL = ACQ = FENCE = true; break;  // win>=1：正确
    }
  } else if (mode == 11) {
    bps = S;  // 正确版 stridedbp（有 worker 抽验）
  } else if (mode == 13) {
    pbs = S;  // 正确版 stridedpub
  } else if (mode == 12) {
    bps = S; pbs = S;
  }
  const bool do_bp = is_gen && BP;
  const bool do_pub = HD;

  // ---------------- producer：block 0..P-1 ----------------
  if (mode != 0 && blockIdx.x < P) {
    const int p = blockIdx.x;
    int b0 = p * bp;
    if (b0 > nb) b0 = nb;
    int b1 = (p + 1) * bp;
    if (b1 > nb) b1 = nb;
    const long long off = (long long)b0 * K;
    const long long len = (long long)(b1 - b0) * K;
    const long long nbp = len / K;  // 本 producer 负责的 batch 数
    unsigned long long pacc = 0;

    // 逐相位计时累加器（lane0）
    unsigned long long t_bp = 0, t_wr = 0, t_gen = 0, t_store = 0, t_pub = 0, t_pw = 0;
    unsigned long long n_timed = 0;

    if (gvar == 3 && !(DW & 3) && (is_gen || smem_only)) {
      // ---------- gvar 3: TMA bulk store（batch 粒度，sb 槽 SMEM 暂存） ----------
      uint32_t* stage = smem;  // sb 个槽，每槽 K*DW 个字
      const uint32_t stage_words = (uint32_t)K * (uint32_t)DW;
      for (long long bj = 0; bj < nbp; ++bj) {
        unsigned long long c0 = 0;
        if (instr && lane == 0 && bj > 0) c0 = clock64();
        // 背压：等本 producer 已消费到「本批 + 预留」条（pure 档 CAP 取满、等价无背压）
        if (do_bp && bps == 1) {
          if (lane == 0) {
            unsigned long long need = (unsigned long long)((bj + 1) * K);
            unsigned long long t = ACQ ? ld_acquire_u64(&tail[p]) : ld_relaxed_u64(&tail[p]);
            long long t0 = clock64();
            while (t + (unsigned long long)CAPp < need) {
              __nanosleep(200);
              t = ACQ ? ld_acquire_u64(&tail[p]) : ld_relaxed_u64(&tail[p]);
              if (clock64() - t0 > 4000000000LL) {
                *abort = 1;
                break;
              }
            }
          }
          __syncthreads();
          if (*abort) break;
        } else if (do_bp && (bj % bps == 0)) {
          // 每 bps 批预留一次：等已消费到 (bj+bps)*K（正确版：一次预留 bps 批容量）
          if (lane == 0) {
            unsigned long long need = (unsigned long long)((bj + bps) * K);
            unsigned long long t = ld_acquire_u64(&tail[p]);
            long long t0 = clock64();
            while (t + (unsigned long long)CAPp < need) {
              __nanosleep(200);
              t = ld_acquire_u64(&tail[p]);
              if (clock64() - t0 > 4000000000LL) {
                *abort = 1;
                break;
              }
            }
          }
          __syncthreads();
          if (*abort) break;
        }
        if (instr && lane == 0 && bj > 0) {
          t_bp += clock64() - c0;
          c0 = clock64();
        }
        // 回收槽：等最老的 store 读完源
        if (lane == 0 && bj >= sb) bulk_wait_read_n(sb - 1);
        __syncthreads();
        if (instr && lane == 0 && bj > 0) {
          t_wr += clock64() - c0;
          c0 = clock64();
        }
        uint32_t* slot = stage + (size_t)(bj % sb) * stage_words;
        // 协作把 K 条描述符生成到 SMEM 槽（内容 = gen_word）；载体 2 直写寄存器，跳过 SMEM 暂存
        if (carrier != 2) {
          for (uint32_t i = lane; i < stage_words; i += nlanes) {
            uint32_t d = i / (uint32_t)DW;
            uint32_t w = i % (uint32_t)DW;
            uint32_t gid = (uint32_t)(off + bj * K + d);
            slot[i] = gen_word(gid, (int)w);
          }
          __syncthreads();
        }
        if (instr && lane == 0 && bj > 0) {
          t_gen += clock64() - c0;
          c0 = clock64();
        }
        uint32_t* dst = ring + (size_t)((size_t)p * CAPp + (size_t)(((bj * K)) % CAPp)) * DW;
        // 发布 head（或跳过 / 摊薄）
        bool pub_now = do_pub && (pbs == 1 || (bj % pbs == pbs - 1) || (bj == nbp - 1));
        if (!smem_only && carrier == 2) {
          // 载体 2：寄存器直写（不落 SMEM；每线程生成 uint4 直接写环，§49 v4 形态）
          const int nv = (int)(stage_words >> 2);
          uint4* d4 = reinterpret_cast<uint4*>(dst);
          for (int u = lane; u < nv; u += nlanes) {
            int i0 = u << 2;
            int d0 = i0 / DW, w0 = i0 % DW;
            uint32_t g0 = (uint32_t)(off + bj * K + d0);
            d4[u] = make_uint4(gen_word(g0, w0), gen_word(g0, w0 + 1), gen_word(g0, w0 + 2),
                               gen_word(g0, w0 + 3));
          }
          if (instr && lane == 0 && bj > 0) {
            t_store += clock64() - c0;
            c0 = clock64();
          }
          __syncthreads();
          if (lane == 0 && pub_now) {
            unsigned long long hv = (unsigned long long)((bj + 1) * K);
            if (WA) __threadfence();
            if (REL)
              st_release_u64(&head[p], hv);
            else
              st_relaxed_u64(&head[p], hv);
          }
          if (instr && lane == 0 && bj > 0) {
            t_pub += clock64() - c0;
            c0 = clock64();
          }
        } else if (!smem_only && carrier == 1) {
          // 载体 1：寄存器 v4 store（全 block 协作把 SMEM 暂存槽整批写到环；普通 global store）
          const int nv = (int)(stage_words >> 2);
          const uint4* s4 = reinterpret_cast<const uint4*>(slot);
          uint4* d4 = reinterpret_cast<uint4*>(dst);
          for (int u = lane; u < nv; u += nlanes) d4[u] = s4[u];
          if (instr && lane == 0 && bj > 0) {
            t_store += clock64() - c0;
            c0 = clock64();
          }
          __syncthreads();  // 发布前全 block 写出可见（寄存器载体固有，TMA 用 bulk 完成等待代替）
          if (lane == 0 && pub_now) {
            unsigned long long hv = (unsigned long long)((bj + 1) * K);
            if (WA) __threadfence();  // 普通 store：用 membar.gpu 代替 bulk 完成等待
            if (REL)
              st_release_u64(&head[p], hv);
            else
              st_relaxed_u64(&head[p], hv);
          }
          if (instr && lane == 0 && bj > 0) {
            t_pub += clock64() - c0;
            c0 = clock64();
          }
        } else if (lane == 0) {
          fence_proxy_async_shared();
          if (!smem_only) {
            bulk_store_1d(dst, slot, stage_words * 4u);
            bulk_commit();
            if (instr && bj > 0) {
              t_store += clock64() - c0;
              c0 = clock64();
            }
            if (pub_now) {
              unsigned long long hv = (unsigned long long)((bj + 1) * K);
              if (WA) bulk_wait_all();
              if (FENCE) fence_proxy_async_global();
              if (REL)
                st_release_u64(&head[p], hv);
              else
                st_relaxed_u64(&head[p], hv);
            }
            if (instr && bj > 0) {
              t_pub += clock64() - c0;
              c0 = clock64();
            }
          }
        }
        // 给消费侧一点工作量把它藏住（PW）
        unsigned int pv = (unsigned int)((long long)bj * K + lane);
        for (int r = 0; r < PW; ++r) pv = pv * 1664525u + 1013904223u;
        pacc += pv;
        __syncthreads();
        if (instr && lane == 0 && bj > 0) t_pw += clock64() - c0;
        if (instr && lane == 0 && bj > 0) n_timed++;
      }
      if (!smem_only && lane == 0) {
        bulk_wait_all();
        // 尾部最后一批总会在 bj==nbp-1 发布（见循环内 pub_now），无需补发
      }
      if (lane == 0 && psink && pacc) atomicAdd(psink, pacc);
      if (instr && lane == 0 && ph && n_timed) {
        unsigned long long base = (unsigned long long)p * 8;
        ph[base + 0] = t_bp / n_timed;
        ph[base + 1] = t_wr / n_timed;
        ph[base + 2] = t_gen / n_timed;
        ph[base + 3] = t_store / n_timed;
        ph[base + 4] = t_pub / n_timed;
        ph[base + 5] = t_pw / n_timed;
        ph[base + 6] = n_timed;
        ph[base + 7] = 0;
      }
      return;
    }

    // ---------- gvar 0/1/2: 寄存器 store（沿用 §48/§49 的 per-step 结构；保留作参照）----------
    const int NWU = DW >> 2;
    const long long step = nlanes;
    for (long long base = 0; base < len; base += step) {
      if (lane == 0) {
        unsigned long long need = (unsigned long long)(base + step);
        unsigned long long t = ld_acquire_u64(&tail[p]);
        long long t0 = clock64();
        while (is_gen && t + (unsigned long long)CAPp < need) {
          __nanosleep(200);
          t = ld_acquire_u64(&tail[p]);
          if (clock64() - t0 > 4000000000LL) {
            *abort = 1;
            break;
          }
        }
        if (*abort) break;
      }
      __syncthreads();
      if (*abort) break;
      if (is_gen) {
        if (gvar == 2 && !(DW & 3)) {
          long long tot_u = (long long)step * NWU;
          for (long long gu = lane; gu < tot_u; gu += nlanes) {
            long long s = base + gu / NWU;
            if (s >= len) break;
            int wu = (int)(gu % NWU);
            uint32_t gid = (uint32_t)(off + s);
            uint32_t* ptr = ring + (size_t)((size_t)p * CAPp + (size_t)(s % CAPp)) * DW + wu * 4;
            uint4 v = make_uint4(gen_word(gid, 4 * wu), gen_word(gid, 4 * wu + 1),
                                 gen_word(gid, 4 * wu + 2), gen_word(gid, 4 * wu + 3));
            *reinterpret_cast<uint4*>(ptr) = v;
          }
        } else {
          long long s = base + lane;
          if (s < len) {
            uint32_t* ptr = ring + (size_t)((size_t)p * CAPp + (size_t)(s % CAPp)) * DW;
            write_desc(ptr, (uint32_t)(off + s), DW, gvar);
          }
        }
      }
      unsigned int pv = (unsigned int)(base + lane);
      for (int r = 0; r < PW; ++r) pv = pv * 1664525u + 1013904223u;
      pacc += pv;
      if (is_gen) asm volatile("fence.proxy.async.global;" ::: "memory");
      __syncthreads();
      if (lane == 0) {
        long long nx = base + step;
        if (nx > len) nx = len;
        st_release_u64(&head[p], (unsigned long long)nx);
      }
    }
    if (lane == 0 && psink && pacc) atomicAdd(psink, pacc);
    return;
  }

  // ---------------- worker ----------------
  const int w = (mode != 0) ? (blockIdx.x - P) : blockIdx.x;
  const int Gw = (mode != 0) ? (gridDim.x - P) : gridDim.x;
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + (size_t)D * K * DW);
  const int nbatch = nb;
  const int bytes_per_batch = K * DW * 4;
  unsigned long long acc = 0;
  // ---- 分配形态（§54 坑②）：0 = 跨步 strided（bb=w+t*Gw，§54 原样）；1 = 连续分块 contig
  //      （bb=w*chunk+t，连续消费同一段，预取只跨相邻 batch，不跨远 owner）。----
  const int wchunk = (nbatch + Gw - 1) / Gw;
  int nt;
  if (wassign == 0) {
    nt = (w < nbatch) ? (nbatch - w + Gw - 1) / Gw : 0;
  } else {
    int b0 = w * wchunk;
    nt = (b0 >= nbatch) ? 0 : (nbatch - b0 < wchunk ? nbatch - b0 : wchunk);
  }
  auto bb_of = [&](int t) -> int { return (wassign == 0) ? (w + t * Gw) : (w * wchunk + t); };

  if (lane == 0)
    for (int i = 0; i < D; ++i) mbar_init(&mbar[i], 1);
  __syncthreads();

  auto owner_of = [&](int bb) -> int {
    int o = (bp > 0) ? (bb / bp) : 0;
    if (o >= P) o = P - 1;
    if (o < 0) o = 0;
    return o;
  };
  auto wait_head = [&](int own, unsigned long long need) -> bool {
    if (mode == 0) return true;
    unsigned long long h = ld_acquire_u64(&head[own]);
    long long wt0 = clock64();
    while (h < need) {
      if (abort && *abort) return false;
      if (clock64() - wt0 > 2000000000LL) {  // worker 侧守门狗：~1.1s，避免死锁时挂死
        if (abort) *abort = 1;
        return false;
      }
      __nanosleep(100);
      h = ld_acquire_u64(&head[own]);
    }
    return true;
  };

  auto issue = [&](int t) -> bool {
    int slot = t % D;
    int bb = bb_of(t);
    int own = owner_of(bb);
    int b0 = own * bp;
    unsigned long long need = (unsigned long long)(bb - b0 + 1) * (unsigned long long)K;
    if (!wait_head(own, need)) return false;
    const uint32_t* src;
    if (is_gen)
      src = ring + (size_t)((size_t)own * CAPp + (size_t)(((bb - b0) * K) % CAPp)) * DW;
    else
      src = static_ring + (size_t)bb * K * DW;
    if (lane == 0) {
      mbar_expect_tx(&mbar[slot], (uint32_t)bytes_per_batch);
      bulk_load_1d(smem + (size_t)slot * K * DW, src, (uint32_t)bytes_per_batch, &mbar[slot]);
    }
    return true;
  };

  bool ok = true;
  for (int t = 0; t < D && t < nt; ++t)
    if (!issue(t)) { ok = false; break; }

  for (int t = 0; t < nt && ok; ++t) {
    int slot = t % D;
    uint32_t phase = (uint32_t)((t / D) & 1u);
    mbar_wait(&mbar[slot], phase);
    const uint32_t* buf = smem + (size_t)slot * K * DW;
    int bb = bb_of(t);
    const uint32_t* dbase = (EDW > 0) ? data + (size_t)bb * K * EDW : data;
    acc += exec_batch(buf, K, DW, dbase, nlanes, lane, EDW, R, bb, df);
    __syncthreads();
    if (is_gen && lane == 0) {
      int own = owner_of(bb);
      atom_add_release_u64(&tail[own], (unsigned long long)K);
    }
    if (t + D < nt)
      if (!issue(t + D)) { ok = false; break; }
  }

  if (acc) atomicAdd(sink, acc);
}

static bool is_gen_mode(int mode);

// =====================================================================================
// host
// =====================================================================================
__global__ void fill_desc_kernel(uint32_t* descs, long long NDESC, int DW) {
  long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= NDESC) return;
  uint32_t* p = descs + id * DW;
  for (int i = 0; i < DW; ++i) p[i] = gen_word((uint32_t)id, i);
}
__global__ void fill_data_kernel(uint32_t* data, long long n, int seed) {
  long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n) return;
  data[id] = mix((uint32_t)seed, (uint32_t)(id & 0xffffffffu)) ^ (uint32_t)(id >> 32);
}

static uint32_t* g_static = nullptr;
static uint32_t* g_ring = nullptr;
static unsigned long long* g_head = nullptr;
static unsigned long long* g_tail = nullptr;
static int* g_abort = nullptr;
static unsigned long long* g_sink = nullptr;
static unsigned long long* g_psink = nullptr;
static unsigned long long* g_ph = nullptr;
static uint32_t* g_data = nullptr;
static long long g_static_alloc = -1;
static int g_dw = -1;
static long long g_ring_alloc = -1;
static long long g_data_alloc = -1;
static int g_edw_filled = -1;
static int g_ctl_p = -1;

static void ensure_static(long long NDESC, int DW) {
  if (g_static_alloc == NDESC && g_dw == DW) return;
  if (g_static) cudaFree(g_static);
  CK(cudaMalloc(&g_static, (size_t)NDESC * DW * 4));
  fill_desc_kernel<<<(int)((NDESC + 255) / 256), 256>>>(g_static, NDESC, DW);
  CK(cudaDeviceSynchronize());
  g_static_alloc = NDESC;
  g_dw = DW;
}
static void ensure_ring(long long CAP, int DW) {
  if (g_ring_alloc == CAP && g_dw == DW) return;
  if (g_ring) cudaFree(g_ring);
  CK(cudaMalloc(&g_ring, (size_t)CAP * DW * 4));
  g_ring_alloc = CAP;
  (void)DW;
}
static void ensure_ctl(int P) {
  if (g_ctl_p >= P && g_head) return;
  if (g_head) { cudaFree(g_head); cudaFree(g_tail); }
  if (P < 1) P = 1;
  CK(cudaMalloc(&g_head, sizeof(unsigned long long) * P));
  CK(cudaMalloc(&g_tail, sizeof(unsigned long long) * P));
  if (!g_sink) {
    CK(cudaMalloc(&g_sink, 8));
    CK(cudaMalloc(&g_psink, 8));
    CK(cudaMalloc(&g_abort, 4));
    CK(cudaMalloc(&g_ph, sizeof(unsigned long long) * 8 * MAXP));
  }
  g_ctl_p = P;
}
static void ensure_data(long long NDESC, int EDW) {
  long long n = (EDW > 0) ? NDESC * (long long)EDW : 1;
  if (n == g_data_alloc && EDW == g_edw_filled) return;
  if (g_data) cudaFree(g_data);
  CK(cudaMalloc(&g_data, (size_t)n * 4));
  fill_data_kernel<<<(int)((n + 255) / 256), 256>>>(g_data, n, 12345);
  CK(cudaDeviceSynchronize());
  g_data_alloc = n;
  g_edw_filled = EDW;
}

struct Res {
  unsigned long long got;
  unsigned long long pgot;
  double ms;
  int aborted;
};

static Res run_scale(int P, int G, long long NDESC, int DW, int EDW, int K, int D, int R, int PW,
                     long long CAPin, int mode, int reps, int T, int gvar, int df = 1,
                     int bpstride = 1, int win = 1, int sb = SB, int instr = 0, int carrier = 0,
                     int wassign = 0) {
  Res res = {0, 0, 0, 0};
  D = (D < 1) ? 1 : D;
  if (D > MAXD) D = MAXD;
  NDESC = (NDESC / K) * K;
  if (P > MAXP) P = MAXP;
  if (P < 1) P = 1;
  const int nb = (int)(NDESC / K);
  const int bp = (nb + P - 1) / P;
  long long CAP = (mode == 0) ? NDESC : CAPin;
  long long CAPp = CAP / P;
  CAPp = (CAPp / K) * K;
  if (CAPp < K) CAPp = K;
  CAP = CAPp * P;
  size_t smem_bytes = (size_t)D * K * DW * 4 + (size_t)D * 8;
  size_t stage_bytes = (size_t)sb * K * DW * 4;  // gvar3 暂存槽
  if (stage_bytes > smem_bytes) smem_bytes = stage_bytes;
  size_t req = smem_bytes;
  if (req < 1024) req = 1024;
  if (req > 48 * 1024)
    CK(cudaFuncSetAttribute(dptd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req));

  ensure_ctl(P);
  ensure_data(NDESC, EDW);
  ensure_static(NDESC, DW);
  if (is_gen_mode(mode) || mode == 6) ensure_ring(CAP, DW);

  int grid;
  if (mode == 0) grid = G;
  else if (mode == 5 || mode == 6 || mode == 7 || mode == 9) grid = P;
  else grid = P + G;  // mode 21 也走 P+G（生产者 + worker 全共驻）

  for (int w = 0; w < 2; ++w) {
    CK(cudaMemset(g_head, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_tail, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_abort, 0, 4));
    dptd_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, gvar,
                                  df, bpstride, win, sb, instr, carrier, wassign, g_static, g_ring, g_head, g_tail,
                                  g_data, g_sink, g_psink, g_abort, g_ph);
    CK(cudaDeviceSynchronize());
  }

  float ms_min = 1e30f;
  unsigned long long got = 0, pgot = 0;
  int aborted = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int r = 0; r < reps; ++r) {
    CK(cudaMemset(g_sink, 0, 8));
    CK(cudaMemset(g_psink, 0, 8));
    CK(cudaMemset(g_head, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_tail, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_abort, 0, 4));
    CK(cudaEventRecord(e0));
    dptd_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, gvar,
                                  df, bpstride, win, sb, instr, carrier, wassign, g_static, g_ring, g_head, g_tail,
                                  g_data, g_sink, g_psink, g_abort, g_ph);
    CK(cudaGetLastError());
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (ms < ms_min) ms_min = ms;
    CK(cudaMemcpy(&got, g_sink, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&pgot, g_psink, 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&aborted, g_abort, 4, cudaMemcpyDeviceToHost));
  }
  res.got = got;
  res.pgot = pgot;
  res.ms = ms_min;
  res.aborted = aborted;
  printf("  P=%-3d G=%-3d EDW=%-5d K=%-4d D=%-3d R=%-6d PW=%-6d CAPp=%-6lld mode=%d gvar=%d win=%d "
         "S=%d  %9.4f ms/rep  got=%llu%s\n",
         P, G, EDW, K, D, R, PW, CAPp, mode, gvar, win, bpstride, res.ms, res.got,
         aborted ? "  ABORT" : "");
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return res;
}

// mode 是否用 gvar3/TMA 写路径
static bool is_gen_mode(int mode) {
  return mode == 1 || mode == 5 || mode == 6 || mode == 8 || mode == 9 || mode == 11 ||
         mode == 12 || mode == 13 || mode == 21;
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  const int NSM = prop.multiProcessorCount;
  printf("device %d: %s  SMs=%d  cc=%d.%d  L2=%d MB\n", dev, prop.name, NSM, prop.major,
         prop.minor, (int)(prop.l2CacheSize >> 20));
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

  const int DW = 88;                // 352 B, 对齐 MPK §6.1
  const int T = 256;
  const long long NDESC = 393216;   // ×352 B ≈ 132 MB ≫ L2（静态数组口径）
  const int K = 32, D_ = 2;
  const double bd = (double)NDESC * DW * 4.0;

  const char* cmd = argc > 1 ? argv[1] : "bench";
  if (strcmp(cmd, "run") == 0) {
    int P = argc > 2 ? atoi(argv[2]) : 1;
    int G = argc > 3 ? atoi(argv[3]) : (NSM - P);
    long long N = argc > 4 ? atoll(argv[4]) : NDESC;
    int dw = argc > 5 ? atoi(argv[5]) : DW;
    int edw = argc > 6 ? atoi(argv[6]) : 256;
    int Kk = argc > 7 ? atoi(argv[7]) : K;
    int D = argc > 8 ? atoi(argv[8]) : D_;
    int R = argc > 9 ? atoi(argv[9]) : 0;
    int PW = argc > 10 ? atoi(argv[10]) : 0;
    long long CAP = argc > 11 ? atoll(argv[11]) : 32768;
    int mode = argc > 12 ? atoi(argv[12]) : 1;
    int reps = argc > 13 ? atoi(argv[13]) : 5;
    int t = argc > 14 ? atoi(argv[14]) : T;
    int gvar = argc > 15 ? atoi(argv[15]) : 0;
    int df = argc > 16 ? atoi(argv[16]) : 1;
    int bpstride = argc > 17 ? atoi(argv[17]) : 1;
    int win = argc > 18 ? atoi(argv[18]) : 1;
    int sb = argc > 19 ? atoi(argv[19]) : SB;
    int instr = argc > 20 ? atoi(argv[20]) : 0;
    int carrier = argc > 21 ? atoi(argv[21]) : 0;
    int wassign = argc > 22 ? atoi(argv[22]) : 0;
    run_scale(P, G, N, dw, edw, Kk, D, R, PW, CAP, mode, reps, t, gvar, df, bpstride, win, sb,
              instr, carrier, wassign);
    return 0;
  }

  // ===== P65-1：写端握手归因的端到端形态（动态生产者 + worker + 数据面）=====
  if (strcmp(cmd, "ebench") == 0) {
    const int Pt = 32;
    const int Gt = NSM - Pt;
    const int De = 1;  // worker 取 task 环深：端到端同步对照用 D=1（D 是 §41/§42 的独立变量；
                       // 且 strided worker 预取 D>1 会跨 producer 子环边界，触发有界环死锁）
    long long CAP = (long long)NSM * K * 4;
    printf("\n设备 SMs=%d；端到端 ebench：P=%d producers + G=%d workers 全共驻，"
           "DW=%d(352B) NDESC=%lld（静态数组=%.1f MB≫L2），K=%d D=%d T=%d\n",
           NSM, Pt, Gt, DW, NDESC, bd / 1e6, K, De, T);

    // §A 正确性：mode21 ser（民主生产者 + worker）vs static（无生产者）
    printf("\n===== §A 正确性：mode21 ser（有 worker）vs static（NDESC=8192, EDW=8, R=0）=====\n");
    {
      long long N = 8192;
      int G = NSM - 4;
      for (int Pv : {1, 2}) {
        long long CAPc = (long long)NSM * K * 4;
        Res st = run_scale(Pv, G, N, DW, 8, K, 1, 0, 0, 0, 0, 1, T, 3);
        Res se = run_scale(Pv, G, N, DW, 8, K, 1, 0, 0, CAPc, 21, 1, T, 3, 1, 1, 1);
        printf("      P=%d  static.got=%llu  ser.got=%llu  %s\n", Pv, st.got, se.got,
               st.got == se.got ? "OK(逐位一致)" : "DIFF");
      }
    }

    // §B headline：端到端归因矩阵 × 数据面大小 EDW
    printf("\n===== §B 端到端归因矩阵（P=%d G=%d，TMA，GB/s = 描述符字节/墙钟）=====\n", Pt, Gt);
    {
      struct V { const char* name; int win; int S; };
      V vs[] = {{"static", 0, 1},   {"ser", 1, 1},       {"nowait", -1, 1},
                {"skipbp", -4, 1},  {"nobp_wa", -16, 1},{"relhead", -7, 1},
                {"reltail", -8, 1}, {"relboth", -9, 1}, {"strall8", -12, 8}};
      const int NV = (int)(sizeof(vs) / sizeof(vs[0]));
      int edws[] = {0, 64, 256, 1024, 4096};
      printf("  %-8s", "EDW");
      for (int i = 0; i < NV; ++i) printf(" %9s", vs[i].name);
      printf("\n");
      for (int edw : edws) {
        printf("  %-8d", edw);
        for (int i = 0; i < NV; ++i) {
          Res r = (vs[i].win == 0)
                      ? run_scale(Pt, Gt, NDESC, DW, edw, K, De, 0, 0, 0, 0, 3, T, 3)
                      : run_scale(Pt, Gt, NDESC, DW, edw, K, De, 0, 0, CAP, 21, 3, T, 3, 1,
                                  vs[i].S, vs[i].win);
          printf(" %9.3f", bd / (r.ms / 1e3) / 1e9);
        }
        printf("\n");
      }
      printf("  （static = 无生产者（worker 只读静态数组）；ser/nobp_wa/skipbp 均为 HD=1 的\"发布\"档，"
             "worker 能等；哨兵档只拆代价、数值不可信）\n");
    }

    // §B2：环容量不 binding（CAP=NDESC，背压永不触发）——把 §53 的**内存序握手税**从**容量耦合**里摘出来
    printf("\n===== §B2 环容量不 binding（CAP=NDESC，backpressure 永不触发）"
           "：隔离内存序握手税（P=%d G=%d EDW=256，GB/s）=====\n", Pt, Gt);
    {
      struct V { const char* name; int win; int S; };
      V vs[] = {{"ser", 1, 1},        {"nowait", -1, 1},     {"skipbp", -4, 1},
                {"nobp_wa", -16, 1},  {"reltail", -8, 1},    {"relhead", -7, 1},
                {"strall8", -12, 8}};
      const int NV = (int)(sizeof(vs) / sizeof(vs[0]));
      printf("  %-9s", "EDW");
      for (int i = 0; i < NV; ++i) printf(" %9s", vs[i].name);
      printf("\n");
      for (int edw : {0, 256}) {
        printf("  %-9d", edw);
        for (int i = 0; i < NV; ++i) {
          Res r = run_scale(Pt, Gt, NDESC, DW, edw, K, De, 0, 0, NDESC, 21, 3, T, 3, 1, vs[i].S,
                            vs[i].win);
          printf(" %9.3f", bd / (r.ms / 1e3) / 1e9);
        }
        printf("\n");
      }
      printf("  （CAP=NDESC ⇒ ser 的背压门永不阻塞，ser vs nobp_wa/skipbp 只差每批的"
             "release/acquire/__syncthreads；此刻才复现 §53 的\"纯写式\"归因）\n");
    }

    // §C 临界路径：扫 P（每格 = pure-write ser / e2e static / e2e ser / e2e nobp_wa，GB/s）
    printf("\n===== §C 临界路径：扫 P（EDW=256；pw_ser / e2e_static / e2e_ser / e2e_nobp_wa，GB/s）=====\n");
    {
      printf("  %-4s %-4s %9s %11s %9s %11s\n", "P", "G", "pw_ser", "e2e_static", "e2e_ser",
             "e2e_nobp_wa");
      for (int Pv : {8, 16, 32, 64}) {
        if (Pv >= NSM) break;
        int Gv = NSM - Pv;
        Res pw = run_scale(Pv, 0, NDESC, DW, 0, K, De, 0, 0, NDESC, 9, 3, T, 3, 1, 1, 1);
        Res es = run_scale(Pv, Gv, NDESC, DW, 256, K, De, 0, 0, CAP, 0, 3, T, 3);
        Res se = run_scale(Pv, Gv, NDESC, DW, 256, K, De, 0, 0, CAP, 21, 3, T, 3, 1, 1, 1);
        Res nw = run_scale(Pv, Gv, NDESC, DW, 256, K, De, 0, 0, CAP, 21, 3, T, 3, 1, 1, -16);
        printf("  %-4d %-4d %9.1f %11.1f %9.1f %11.1f\n", Pv, Gv, bd / (pw.ms / 1e3) / 1e9,
               bd / (es.ms / 1e3) / 1e9, bd / (se.ms / 1e3) / 1e9, bd / (nw.ms / 1e3) / 1e9);
      }
    }

    // §D 载体 × 端到端（P=32 G=100 EDW=256；ser / nobp_wa，GB/s）
    printf("\n===== §D 载体 × 端到端（P=%d G=%d EDW=256，GB/s）=====\n", Pt, Gt);
    {
      struct C { const char* n; int c; };
      C cs[] = {{"tma", 0}, {"regS(smem)", 1}, {"regD(reg)", 2}};
      printf("  %-12s %10s %10s %8s\n", "carrier", "e2e_ser", "e2e_nobp_wa", "nwa/ser");
      for (auto& c : cs) {
        Res se = run_scale(Pt, Gt, NDESC, DW, 256, K, De, 0, 0, CAP, 21, 3, T, 3, 1, 1, 1, SB, 0, c.c);
        Res nw = run_scale(Pt, Gt, NDESC, DW, 256, K, De, 0, 0, CAP, 21, 3, T, 3, 1, 1, -16, SB, 0, c.c);
        double bs = bd / (se.ms / 1e3) / 1e9, bn = bd / (nw.ms / 1e3) / 1e9;
        printf("  %-12s %10.1f %10.1f %8.3f\n", c.n, bs, bn, bn / bs);
      }
    }

    if (g_static) cudaFree(g_static);
    if (g_ring) cudaFree(g_ring);
    if (g_data) cudaFree(g_data);
    if (g_head) { cudaFree(g_head); cudaFree(g_tail); }
    if (g_ph) cudaFree(g_ph);
    cudaFree(g_sink);
    cudaFree(g_psink);
    cudaFree(g_abort);
    return 0;
  }

  // ===== P66-1：worker 取 task 的预取深度 D × 分配形态（端到端）=====
  if (strcmp(cmd, "wbench") == 0) {
    const int Pt = 32;
    const int Gt = NSM - Pt;
    printf("\n设备 SMs=%d；wbench：P=%d producers + G=%d workers 全共驻，DW=%d(352B) NDESC=%lld"
           "（静态数组=%.1f MB≫L2），K=%d T=%d\n",
           NSM, Pt, Gt, DW, NDESC, bd / 1e6, K, T);
    const long long CAPtight = (long long)16 * K * Pt;  // 每 producer 16 批容量

    // §A 正确性：mode21 ser × {strided,contig} × D（N=8192, EDW=8）逐位对 static
    printf("\n===== §A 正确性：mode21 ser × {strided,contig} × D（N=8192, EDW=8, R=0）=====\n");
    {
      long long N = 8192;
      int G = NSM - 4;
      long long CAPc = (long long)NSM * K * 4;
      Res st = run_scale(1, G, N, DW, 8, K, 1, 0, 0, 0, 0, 1, T, 3, 1);
      printf("      static.got=%llu\n", st.got);
      for (int as : {0, 1}) {
        printf("      assign=%-7s:", as ? "contig" : "strided");
        for (int Dv : {1, 2, 4}) {
          Res r = run_scale(1, G, N, DW, 8, K, Dv, 0, 0, CAPc, 21, 1, T, 3, 1, 1, 1, SB, 0, 0, as);
          printf("  D%d=%s", Dv, r.got == st.got ? "==" : "!=");
        }
        printf("\n");
      }
    }

    // §B 预取深度 × 分配（环容量不 binding CAP=NDESC，背压永不触发）
    printf("\n===== §B 预取深度 × 分配（CAP=NDESC 不 binding，P=%d G=%d，GB/s）=====\n", Pt, Gt);
    {
      printf("  %-8s %-8s", "EDW", "assign");
      for (int Dv : {1, 2, 4, 8}) printf(" %8s", "D");
      printf("\n");
      for (int edw : {0, 256}) {
        for (int as : {0, 1}) {
          printf("  %-8d %-8s", edw, as ? "contig" : "strided");
          for (int Dv : {1, 2, 4, 8}) {
            Res r = run_scale(Pt, Gt, NDESC, DW, edw, K, Dv, 0, 0, NDESC, 21, 3, T, 3, 1, 1, 1, SB, 0,
                              0, as);
            printf(" %8.1f", bd / (r.ms / 1e3) / 1e9);
          }
          printf("\n");
        }
      }
      printf("  （每格 = 描述符字节/墙钟 GB/s；ser 正确握手；D = worker 取 task 环深/预取深度）\n");
    }

    // §C 预取深度 × 分配（紧环 CAPp=16 批，背压真正触发）——死锁/正确性边界随分配变化
    printf("\n===== §C 预取深度 × 分配（紧环 CAPp=16 批，EDW=256）=====\n");
    {
      unsigned long long ref256 =
          run_scale(Pt, Gt, NDESC, DW, 256, K, 1, 0, 0, 0, 0, 1, T, 3, 1).got;
      printf("  static(EDW=256) got=%llu\n", ref256);
      printf("  %-8s %-8s", "EDW", "assign");
      for (int Dv : {1, 2, 4, 8}) printf(" %14s", "D");
      printf("\n");
      for (int as : {0, 1}) {
        printf("  %-8d %-8s", 256, as ? "contig" : "strided");
        for (int Dv : {1, 2, 4, 8}) {
          Res r = run_scale(Pt, Gt, NDESC, DW, 256, K, Dv, 0, 0, CAPtight, 21, 3, T, 3, 1, 1, 1, SB,
                            0, 0, as);
          char buf[32];
          if (r.aborted)
            snprintf(buf, sizeof(buf), "ABORT");
          else if (r.got != ref256)
            snprintf(buf, sizeof(buf), "%.1f WRONG", bd / (r.ms / 1e3) / 1e9);
          else
            snprintf(buf, sizeof(buf), "%.1f OK", bd / (r.ms / 1e3) / 1e9);
          printf(" %14s", buf);
        }
        printf("\n");
      }
      printf("  （strided D=1 正确握手；D>1 预取跨 producer 子环 → 死锁；contig D=1 在 producer 被拆到"
             "多个 worker 时数值 WRONG，见 §E 机制）\n");
    }

    // §E 机制隔离：per-producer 的前缀 tail 要求「每个 producer 被**单个** worker 顺序消费」。
    //     contig 下每个 worker 覆盖连续 chunk；当 producer 边界与 worker 边界不对齐（chunk≠bp）时，
    //     一个 producer 被相邻两个 worker 并发消费——高 batch 可先被消费 → tail 单调推进但不再代表
    //     「已消费前缀」→ producer 覆写仍在飞的 TMA/未消费 slot。chunk==bp（本机 G=P）时对齐 → OK。
    printf("\n===== §E 机制隔离（紧环 CAPp=16 批，EDW=256，D=1）：contig 正确性 × 边界是否对齐 =====\n");
    {
      unsigned long long ref = run_scale(Pt, Gt, NDESC, DW, 256, K, 1, 0, 0, 0, 0, 1, T, 3, 1).got;
      printf("  static(EDW=256) got=%llu\n", ref);
      struct S { int P; int G; const char* tag; };
      S ss[] = {{32, 100, "split(chunk<bp)"}, {100, 32, "split(chunk>bp)"}, {66, 66, "aligned(chunk==bp)"}};
      printf("  %-5s %-5s %-20s %14s %14s\n", "P", "G", "regime", "contig", "strided");
      for (auto& s : ss) {
        long long CAPt = (long long)16 * K * s.P;
        Res c = run_scale(s.P, s.G, NDESC, DW, 256, K, 1, 0, 0, CAPt, 21, 1, T, 3, 1, 1, 1, SB, 0, 0, 1);
        Res t = run_scale(s.P, s.G, NDESC, DW, 256, K, 1, 0, 0, CAPt, 21, 1, T, 3, 1, 1, 1, SB, 0, 0, 0);
        printf("  %-5d %-5d %-20s %14s %14s\n", s.P, s.G, s.tag,
               c.aborted ? "ABORT" : (c.got == ref ? "OK" : "WRONG"),
               t.aborted ? "ABORT" : (t.got == ref ? "OK" : "WRONG"));
      }
      printf("  （chunk=nb/Gw，bp=nb/P；producer 被拆 ⟺ chunk≠bp ⟹ contig 错；chunk==bp 对齐 ⟹ OK）\n");
    }

    // §D worker 侧地板（无生产者 mode0 = worker 只读静态数组）× D × 分配
    printf("\n===== §D worker 侧地板（无生产者 mode0，EDW=256，GB/s）=====\n");
    {
      printf("  %-8s %-8s", "EDW", "assign");
      for (int Dv : {1, 2, 4, 8}) printf(" %8s", "D");
      printf("\n");
      for (int edw : {0, 256}) {
        for (int as : {0, 1}) {
          printf("  %-8d %-8s", edw, as ? "contig" : "strided");
          for (int Dv : {1, 2, 4, 8}) {
            Res r = run_scale(Pt, Gt, NDESC, DW, edw, K, Dv, 0, 0, 0, 0, 3, T, 3, 1, 1, 1, SB, 0, 0,
                              as);
            printf(" %8.1f", bd / (r.ms / 1e3) / 1e9);
          }
          printf("\n");
        }
      }
    }

    if (g_static) cudaFree(g_static);
    if (g_ring) cudaFree(g_ring);
    if (g_data) cudaFree(g_data);
    if (g_head) { cudaFree(g_head); cudaFree(g_tail); }
    if (g_ph) cudaFree(g_ph);
    cudaFree(g_sink);
    cudaFree(g_psink);
    cudaFree(g_abort);
    return 0;
  }

  // ===== P64-1：写端握手归因 × 写载体（TMA bulk store vs 寄存器 v4 store）=====
  if (strcmp(cmd, "cbench") == 0) {
    struct V { const char* name; int win; int S; };
    V vs[] = {{"ser", 1, 1},        {"nowait", -1, 1},     {"skipbp", -4, 1},
              {"skiphd", -5, 1},    {"nobp", -6, 1},       {"relhead", -7, 1},
              {"reltail", -8, 1},   {"relboth", -9, 1},    {"nobp_wa", -16, 1},
              {"wa_only", -17, 1},  {"hd_nofence", -15, 1},"stridedall8", -12, 8};
    const int NV = (int)(sizeof(vs) / sizeof(vs[0]));
    const int P = 128;

    // §A 正确性：carrier1（寄存器 v4 写环）正确档 vs static（有 worker）
    printf("\n===== §A 正确性：carrier1 正确档 vs static（有 worker，EDW=8）=====\n");
    {
      long long N = 8192;
      int G = NSM - 4;
      long long CAP = (long long)NSM * K * 4;
      unsigned long long ref = 0;
      for (int Pv : {1, 2}) {
        Res st = run_scale(Pv, G, N, DW, 8, K, 1, 0, 0, 0, 0, 1, T, 3);
        if (Pv == 1) ref = st.got;
        Res g0 = run_scale(Pv, G, N, DW, 8, K, 1, 0, 0, CAP, 1, 1, T, 3, 1, 1, 1, SB, 0, 0);
        Res g1 = run_scale(Pv, G, N, DW, 8, K, 1, 0, 0, CAP, 1, 1, T, 3, 1, 1, 1, SB, 0, 1);
        Res g2 = run_scale(Pv, G, N, DW, 8, K, 1, 0, 0, CAP, 1, 1, T, 3, 1, 1, 1, SB, 0, 2);
        Res s1 = run_scale(Pv, G, N, DW, 8, K, 1, 0, 0, CAP, 11, 1, T, 3, 1, 4, 1, SB, 0, 1);
        Res s2 = run_scale(Pv, G, N, DW, 8, K, 1, 0, 0, CAP, 11, 1, T, 3, 1, 4, 1, SB, 0, 2);
        printf("      P=%d  static%s  c0-ser%s  c1-ser%s  c2-ser%s  c1-sbp%s  c2-sbp%s\n", Pv,
               st.got == ref ? "==" : "!=", g0.got == ref ? "==" : "!=", g1.got == ref ? "==" : "!=",
               g2.got == ref ? "==" : "!=", s1.got == ref ? "==" : "!=", s2.got == ref ? "==" : "!=");
      }
    }

    // §B headline：三载体的归因矩阵（纯写 P=128，同 D=2；GB/s）
    printf("\n===== §B 归因矩阵 × 载体（纯写 P=128，GB/s；BD=%.1f MB）=====\n", bd / 1e6);
    {
      printf("  %-12s | %8s %7s | %8s %7s | %8s %7s\n", "variant", "tma", "vs_ser", "regS",
             "vs_ser", "regD", "vs_ser");
      double s0 = 0, s1 = 0, s2 = 0;
      for (int i = 0; i < NV; ++i) {
        Res t = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, vs[i].S, vs[i].win,
                          SB, 0, 0);
        Res r = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, vs[i].S, vs[i].win,
                          SB, 0, 1);
        Res q = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, vs[i].S, vs[i].win,
                          SB, 0, 2);
        double bt = bd / (t.ms / 1e3) / 1e9, br = bd / (r.ms / 1e3) / 1e9,
               bq = bd / (q.ms / 1e3) / 1e9;
        if (vs[i].win == 1) { s0 = bt; s1 = br; s2 = bq; }
        printf("  %-12s | %8.1f %7.3f | %8.1f %7.3f | %8.1f %7.3f\n", vs[i].name, bt,
               s0 > 0 ? bt / s0 : 0.0, br, s1 > 0 ? br / s1 : 0.0, bq, s2 > 0 ? bq / s2 : 0.0);
      }
    }

    // §B2 地板与相对代价汇总（nobp=store-only 地板）
    printf("\n===== §B2 地板与代价（纯写 P=128，GB/s）=====\n");
    {
      struct C { const char* n; int c; };
      C cs[] = {{"tma", 0}, {"regS(smem)", 1}, {"regD(reg)", 2}};
      printf("  %-12s %10s %10s %10s %10s %8s\n", "carrier", "floor(nobp)", "ser", "relhead",
             "ser-nobp%", "floor/ser");
      for (auto& c : cs) {
        Res f = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, -6, SB, 0, c.c);
        Res s = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, 1, SB, 0, c.c);
        Res h = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, -7, SB, 0, c.c);
        double bf = bd / (f.ms / 1e3) / 1e9, bs = bd / (s.ms / 1e3) / 1e9,
               bh = bd / (h.ms / 1e3) / 1e9;
        printf("  %-12s %10.1f %10.1f %10.1f %10.1f %8.3f\n", c.n, bf, bs, bh,
               (bf - bs) / bf * 100.0, bf / bs);
      }
    }

    // §C × P：三载体的 ser / nobp(地板)
    printf("\n===== §C × P（纯写 GB/s；每格 = ser / nobp）=====\n");
    {
      printf("  %-4s | %16s | %16s | %16s\n", "P", "tma", "regS(smem)", "regD(reg)");
      for (int Pv : {16, 32, 64, 128}) {
        if (Pv > NSM) break;
        double v[3][2];
        for (int c = 0; c < 3; ++c) {
          Res s = run_scale(Pv, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, 1, SB, 0, c);
          Res n = run_scale(Pv, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, -6, SB, 0, c);
          v[c][0] = bd / (s.ms / 1e3) / 1e9;
          v[c][1] = bd / (n.ms / 1e3) / 1e9;
        }
        printf("  %-4d | %7.1f/%7.1f | %7.1f/%7.1f | %7.1f/%7.1f\n", Pv, v[0][0], v[0][1], v[1][0],
               v[1][1], v[2][0], v[2][1]);
      }
    }

    // §D 摊薄步长 S（stridedall）三载体
    printf("\n===== §D 摊薄 stridedall 步长 S（纯写 P=128，GB/s）=====\n");
    {
      printf("  %-4s %13s %13s %13s\n", "S", "tma", "regS", "regD");
      for (int S : {1, 2, 4, 8, 16, 32}) {
        Res t = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, S, -12, SB, 0, 0);
        Res r = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, S, -12, SB, 0, 1);
        Res q = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, S, -12, SB, 0, 2);
        printf("  %-4d %13.1f %13.1f %13.1f\n", S, bd / (t.ms / 1e3) / 1e9,
               bd / (r.ms / 1e3) / 1e9, bd / (q.ms / 1e3) / 1e9);
      }
    }

    // §E 逐相位（纯写 P=128，lane0 clock64，cycles/batch）：三载体 ser / nobp(地板) / relhead
    printf("\n===== §E 逐相位（纯写 P=128，cycles/batch）=====\n");
    {
      struct W { const char* name; int win; };
      W ws[] = {{"ser", 1}, {"nobp", -6}, {"relhead", -7}};
      printf("  %-5s %-9s %8s %7s %7s %8s %7s %7s %7s\n", "car", "variant", "bp", "wr", "gen",
             "store", "pub", "pw", "sum");
      for (int car : {0, 1, 2}) {
        for (auto& v : ws) {
          run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 3, T, 3, 1, 1, v.win, SB, 1, car);
          unsigned long long hbuf[8 * MAXP];
          CK(cudaMemcpy(hbuf, g_ph, sizeof(unsigned long long) * 8 * MAXP, cudaMemcpyDeviceToHost));
          unsigned long long s0 = 0;
          for (int i = 0; i < 6; ++i) s0 += hbuf[i];
          printf("  %-5s %-9s %8llu %7llu %7llu %8llu %7llu %7llu %7llu\n",
                 car == 0 ? "tma" : (car == 1 ? "regS" : "regD"), v.name, hbuf[0], hbuf[1], hbuf[2],
                 hbuf[3], hbuf[4], hbuf[5], s0);
        }
      }
    }

    if (g_static) cudaFree(g_static);
    if (g_ring) cudaFree(g_ring);
    if (g_data) cudaFree(g_data);
    if (g_head) { cudaFree(g_head); cudaFree(g_tail); }
    if (g_ph) cudaFree(g_ph);
    cudaFree(g_sink);
    cudaFree(g_psink);
    cudaFree(g_abort);
    return 0;
  }

  printf("全共驻 grid=P+G=%d；DW=%d(352B) NDESC=%lld 静态数组=%.1f MB；K=%d D=%d T=%d\n", NSM, DW,
         NDESC, bd / 1e6, K, D_, T);

  // §A 正确性：正确档（serialized / pipelined / 摊薄背压 mode11 / 摊薄发布 mode13）逐位一致
  printf("\n===== §A 正确性：正确档 vs static（NDESC=8192, EDW=8, R=0, 有 worker）=====\n");
  {
    long long N = 8192;
    unsigned long long ref = 0;
    int G = NSM - 4;
    for (int P : {1, 2}) {
      long long CAP = (long long)NSM * K * 4;
      long long lead = (long long)4 * K;
      if (CAP < lead) CAP = lead;
      Res st = run_scale(P, G, N, DW, 8, K, 1, 0, 0, 0, 0, 1, T, 3);  // static 参照
      if (P == 1) ref = st.got;
      Res g1 = run_scale(P, G, N, DW, 8, K, 1, 0, 0, CAP, 1, 1, T, 3, 1, 1, 1);   // serialized gen
      Res gb = run_scale(P, G, N, DW, 8, K, 1, 0, 0, CAP, 11, 1, T, 3, 1, 4, 1);  // stridedbp S=4
      Res gp = run_scale(P, G, N, DW, 8, K, 1, 0, 0, CAP, 13, 1, T, 3, 1, 4, 1);  // stridedpub S=4
      Res ga = run_scale(P, G, N, DW, 8, K, 1, 0, 0, CAP, 12, 1, T, 3, 1, 4, 1);  // stridedall S=4
      printf("      P=%d  static%s  ser%s  stridedbp%s  stridedpub%s  stridedall%s\n", P,
             st.got == ref ? "==" : "!=", g1.got == ref ? "==" : "!=",
             gb.got == ref ? "==" : "!=", gp.got == ref ? "==" : "!=",
             ga.got == ref ? "==" : "!=");
    }
  }

  // §B 归因矩阵（纯写，P=128，无 worker）：哨兵档定位「贵在哪一维」
  printf("\n===== §B 归因矩阵（纯写 mode5/9，无 worker，P=128，GB/s；bd=138.4MB）=====\n");
  {
    int P = 128;
    struct V { const char* name; int win; int S; };
    V vs[] = {{"ser", 1, 1},         {"nowait", -1, 1},      {"skipbp", -4, 1},
              {"skiphd", -5, 1},     {"nobp", -6, 1},        {"relhead", -7, 1},
              {"reltail", -8, 1},    {"relboth", -9, 1},     {"stridedbp", -10, 4},
              {"stridedpub", -11, 4},{"stridedall", -12, 4},{"nowa_relhead", -2, 1},
              {"nobp_wa", -16, 1},   {"wa_only", -17, 1},    {"hd_nofence", -15, 1},
              {"nowa_strpub", -13, 4}};
    printf("  %-12s %11s %11s %11s\n", "variant", "ms", "GB/s", "vs_ser");
    double serbw = 0;
    for (auto& v : vs) {
      Res r = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, v.S, v.win, SB);
      double bw = bd / (r.ms / 1e3) / 1e9;
      if (v.win == 1) serbw = bw;
      printf("  %-12s %11.4f %11.1f %11.3f\n", v.name, r.ms, bw, bw / serbw);
    }
  }

  // §B2 摊薄步长 S 的标度（stridedbp / stridedpub / stridedall）
  printf("\n===== §B2 摊薄步长 S 标度（纯写 P=128，GB/s）=====\n");
  {
    int P = 128;
    printf("  %-4s %11s %11s %11s %11s %11s\n", "S", "stridedbp", "stridedpub", "stridedall",
           "nowa_spub", "nowa_sbp");
    for (int S : {1, 2, 4, 8, 16, 32}) {
      if (S > 32) break;
      Res a = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, S, -10, SB);
      Res b = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, S, -11, SB);
      Res c = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, S, -12, SB);
      Res d = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, S, -13, SB);
      Res e = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, S, -14, SB);
      printf("  %-4d %11.1f %11.1f %11.1f %11.1f %11.1f\n", S, bd / (a.ms / 1e3) / 1e9,
             bd / (b.ms / 1e3) / 1e9, bd / (c.ms / 1e3) / 1e9, bd / (d.ms / 1e3) / 1e9,
             bd / (e.ms / 1e3) / 1e9);
    }
  }

  // §C 归因矩阵 × P（看不同 P 下结论是否一致）
  printf("\n===== §C 归因矩阵 × P（纯写，GB/s）=====\n");
  {
    printf("  %-4s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", "P", "ser", "nowait", "skipbp",
           "skiphd", "relhead", "reltail", "relboth", "strall(S8)", "nowa_rhead");
    for (int P : {16, 32, 64, 128}) {
      if (P > NSM) break;
      Res se = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 5, T, 3);
      Res nw = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, -1, SB);
      Res bp = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, -4, SB);
      Res hd = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, -5, SB);
      Res rh = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, -7, SB);
      Res rt = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, -8, SB);
      Res rb = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, -9, SB);
      Res al = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 8, -12, SB);
      Res nr = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 5, T, 3, 1, 1, -2, SB);
      double d = 1e3;
      printf("  %-4d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", P,
             bd / (se.ms / d) / 1e9, bd / (nw.ms / d) / 1e9, bd / (bp.ms / d) / 1e9,
             bd / (hd.ms / d) / 1e9, bd / (rh.ms / d) / 1e9, bd / (rt.ms / d) / 1e9,
             bd / (rb.ms / d) / 1e9, bd / (al.ms / d) / 1e9, bd / (nr.ms / d) / 1e9);
    }
  }

  // §D 逐相位计时（lane0 的 clock64 累加，P=128 纯写）：ser vs skipbp vs skiphd vs relhead
  printf("\n===== §D 逐相位计时（P=128 纯写，lane0 clock64，cycles/batch）=====\n");
  {
    int P = 128;
    struct V { const char* name; int win; };
    V vs[] = {{"ser", 1}, {"nowait", -1}, {"skipbp", -4}, {"skiphd", -5}, {"relhead", -7}};
    printf("  %-10s %9s %9s %9s %9s %9s %9s %9s\n", "variant", "bp", "wr", "gen", "store", "pub",
           "pw", "sum");
    for (auto& v : vs) {
      run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 3, T, 3, 1, 1, v.win, SB, 1);
      unsigned long long hbuf[8 * MAXP];
      CK(cudaMemcpy(hbuf, g_ph, sizeof(unsigned long long) * 8 * MAXP, cudaMemcpyDeviceToHost));
      // 取有代表性的一个 producer（p=0）
      unsigned long long s0 = 0;
      for (int i = 0; i < 6; ++i) s0 += hbuf[i];
      printf("  %-10s %9llu %9llu %9llu %9llu %9llu %9llu %9llu\n", v.name, hbuf[0], hbuf[1],
             hbuf[2], hbuf[3], hbuf[4], hbuf[5], s0);
    }
  }

  if (g_static) cudaFree(g_static);
  if (g_ring) cudaFree(g_ring);
  if (g_data) cudaFree(g_data);
  if (g_head) { cudaFree(g_head); cudaFree(g_tail); }
  if (g_ph) cudaFree(g_ph);
  cudaFree(g_sink);
  cudaFree(g_psink);
  cudaFree(g_abort);
  return 0;
}
