// dyn_producer_tma_pipe.cu — MegaKernel 专题 P62-1 小实验：
//   **动态生产者 TMA store 的「正确握手 + 流水」**——补齐 §50.7（P61-1）明留的「最值得做的下一条尾巴」。
//
// 动机 / 接续（补掉 §50.7 明留的尾巴）：
//   §50（P61-1）把动态生产者「现场生成描述符」的写载体扩到最宽的 **TMA bulk store**
//   （`cp.async.bulk.global.shared::cta`，SMEM→global），结论：**TMA store 追不平读带宽**——正确握手
//   （serialized，每批 `cp.async.bulk.wait_group 0`）只有 **472 GB/s**，even 大块 rolling 也仅 ~828、
//   不及 16 B 寄存器 v4/coop。但它自陈那条 472 来自**过保守的握手**：
//    - 「**『正确握手 + 流水』的组合未测**：一个**流水但正确**的形态（FIFO commit 后、发布第 j 批 head 前
//      只 `cp.async.bulk.wait_group (win-1)` 保证最老批完成、`head` 落后 `win` 批发布）**未做**——
//      理论可把 serialized 的 472 拉回 rolling 的 ~722–828 且保持正确。这是最值得做的下一条尾巴。」
//   本节把这条尾巴做出来。**唯一新变量 = 发布握手**：serialized（win=1，每批全 drain）vs
//   pipelined（win≥2，FIFO commit + `wait_group (win-1)`，head 滞后 win 批），载体固定 TMA bulk store。
//
//   三个问题：
//   **Q1（§A）** 流水握手写出的 payload 是否仍与寄存器写逐位一致（窗口化后跨 proxy 可见性是否仍正确）？
//   **Q2（§B，headline）** 流水握手把 `gen` 相对 static 的代价从 serialized 的 206× 压到多少？窗口 `win` 多大够？
//   **Q3（§C）** 纯写吞吐：ser 的 472 → 流水能否追回 rolling 的 722~828？窗口 `win` 的标度？
//
// 真实锚点（mirage 快照 `f9eb70c`，与 docs/06–docs/09 同快照）：
//   - 与 §50 同：MPK 的 worker **取 task 用 TMA bulk 1D**；producer 侧（scheduler）**不现场生成**——
//     `execute_scheduler`（`persistent_kernel.cuh:1254-1523`）只做「事件→task id」映射入队，payload 来自
//     编译期一次性 emit 的 `FullTaskDesc` 数组（`runtime.cc`）。本节量的是 MPK **没有采用**的路线。
//   - MPK 的队列协议本身就是「**relaxed 写数据 + 单条 release 推游标**」，靠**批量/环形**摊薄一次
//     release/acquire（`docs/06` §5.3 / 本专题 P47-1 §36）；本节把同样的「批量摊薄 + 窗口流水」用在
//     TMA store 的**跨 proxy 正确握手**上。
//
// 机制（与 §50 同骨架，唯一新变量 = 发布握手）：
//   grid = P + G：block `0..P-1` = producer（P 个 SM），其余 G = worker。batch 按连续区间分给 P 个
//   producer；环按 producer 分区（`CAPp=CAP/P` 槽、自己的 `head[p]/tail[p]`、P 条独立 SPSC 子环）。
//   写载体固定 **gvar 3 = TMA bulk store**（batch 粒度协作生成到 SMEM 暂存槽 → `cp.async.bulk` 整批落环）。
//   发布握手两档：
//     - **serialized（win=1）** = §50 的 `gen`：每批 store 后 `cp.async.bulk.wait_group 0`（全 drain）→
//       `fence.proxy.async.global` → `st.release head=(bj+1)*K`。**过保守**（472 GB/s）。
//     - **pipelined（win≥2，本节新增）**：每批 `bulk_commit` 成一组；发布第 `bj-win+1` 批 head 前只
//       `cp.async.bulk.wait_group (win-1)`（最老一组完成）→ `fence.proxy.async.global` →
//       `st.release head=(bj-win+2)*K`。**head 落后 win 批**，但仍是**正确**握手（发布时该批 store 已完成）。
//       SMEM 暂存槽 `SB=4` 个，`W=min(SB,win)`。
//   `mode 8` = gen-pipe（有 worker、背压、窗口化握手）；`mode 9` = pure-pipe（只 launch P 个 producer、
//   无 worker/背压、窗口化握手，量纯写）；`mode 1`/`5` = §50 的 serialized gen/pure。
//
// 正确握手关键（与 §50 同）：跨 SM 的 TMA store（async proxy 写）要能被另一 SM 的 TMA load（async proxy 读）
//   看到，必须 `cp.async.bulk.wait_group`（写完成，不带 `.read`）→ `fence.proxy.async.global` → 再
//   `st.release` 发布。窗口化只是**推迟发布**、不是跳过完成等待，故仍正确。
//
// 安全：大 CAP 关掉背压只量 producer 吞吐；producer 有 `clock64` 无进度守门狗防死锁挂死，脚本再带 `timeout`。
//
// 编译：nvcc -O3 -arch=sm_90a src/dyn_producer_tma_pipe.cu -o /tmp/dyn_producer_tma_pipe
// 用法：./dyn_producer_tma_pipe run <P> <G> <NDESC> <DW> <EDW> <K> <D> <R> <PW> <CAP> <mode> <reps> [T] [gvar] [df] [tw] [win]

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

// ---- 内存序原语（对齐 MPK 队列协议；快照 mpk_atoms.cuh:17-96） ----
__device__ __forceinline__ unsigned long long ld_acquire_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_u64(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.gpu.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long atom_add_release_u64(unsigned long long* p,
                                                                   unsigned long long v) {
  unsigned long long old;
  asm volatile("atom.add.release.gpu.u64 %0, [%1], %2;" : "=l"(old) : "l"(p), "l"(v)
               : "memory");
  return old;
}

// ---- 硬件 mbarrier + TMA bulk load（worker 取 task） ----
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
    case 12: asm volatile("cp.async.bulk.wait_group.read 12;\n" ::: "memory"); break;
    case 14: asm volatile("cp.async.bulk.wait_group.read 14;\n" ::: "memory"); break;
    default: asm volatile("cp.async.bulk.wait_group.read 15;\n" ::: "memory"); break;
  }
}
__device__ __forceinline__ void bulk_wait_all() {  // 等全部 store 完成（写可见）
  asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}
// 等至多 n 组 store 未完成（n=0 等价 bulk_wait_all）。
__device__ __forceinline__ void bulk_wait_n(int n) {
  switch (n) {
    case 0: asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); break;
    case 1: asm volatile("cp.async.bulk.wait_group 1;\n" ::: "memory"); break;
    case 2: asm volatile("cp.async.bulk.wait_group 2;\n" ::: "memory"); break;
    case 3: asm volatile("cp.async.bulk.wait_group 3;\n" ::: "memory"); break;
    case 4: asm volatile("cp.async.bulk.wait_group 4;\n" ::: "memory"); break;
    case 5: asm volatile("cp.async.bulk.wait_group 5;\n" ::: "memory"); break;
    case 6: asm volatile("cp.async.bulk.wait_group 6;\n" ::: "memory"); break;
    case 7: asm volatile("cp.async.bulk.wait_group 7;\n" ::: "memory"); break;
    case 8: asm volatile("cp.async.bulk.wait_group 8;\n" ::: "memory"); break;
    case 12: asm volatile("cp.async.bulk.wait_group 12;\n" ::: "memory"); break;
    case 14: asm volatile("cp.async.bulk.wait_group 14;\n" ::: "memory"); break;
    default: asm volatile("cp.async.bulk.wait_group 15;\n" ::: "memory"); break;
  }
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

// =====================================================================================
// 主 kernel：P 个 producer（block 0..P-1）+ G 个 worker（其余 block）
//   mode 0 = 静态预写（读大数组 static_ring）
//   mode 1 = gen（producer 现场生成 payload，serialized 握手，win=1）
//   mode 2 = sched（payload 预写大数组，producer 只推 head[p]）
//   mode 5 = pure（只有 P 个 producer，无 worker/背压，serialized 握手，量纯写吞吐）
//   mode 6 = sgen（只有 P 个 producer，只生成到 SMEM、不落 global，隔离 SMEM 暂存代价；仅 gvar3）
//   mode 7 = pure-big（纯写，每 NB 个 batch 一次大块 store，滚动 commit）
//   mode 8 = gen-pipe（有 worker/背压，窗口化正确握手，win≥2；win=1 退化为 mode 1）
//   mode 9 = pure-pipe（只有 P 个 producer，窗口化正确握手，量纯写）
// =====================================================================================
__global__ void dptp_kernel(int P, int NDESC, int DW, int EDW, int K, int D, int R, int PW,
                            int CAP, int mode, int bp, int gvar, int df, int tw, int win, int sb,
                            const uint32_t* __restrict__ static_ring,
                            uint32_t* __restrict__ ring, unsigned long long* __restrict__ head,
                            unsigned long long* __restrict__ tail,
                            const uint32_t* __restrict__ data,
                            unsigned long long* __restrict__ sink,
                            unsigned long long* __restrict__ psink, int* __restrict__ abort) {
  extern __shared__ __align__(16) uint32_t smem[];
  const int lane = threadIdx.x;
  const int nlanes = blockDim.x;
  const int nb = NDESC / K;
  const int CAPp = (mode == 0) ? 0 : (CAP / P);  // 每 producer 子环槽数（K 的倍数）
  const bool is_gen = (mode == 1 || mode == 5 || mode == 8 || mode == 9);
  const bool smem_only = (mode == 6);
  const bool windowed = (mode == 8 || mode == 9);
  const bool nowait = windowed && (win <= -1);  // 哨兵：跳过完成等待只推 head（**不正确**，仅测上界）
  const bool nofence = windowed && (win <= -2);  // 哨兵：连 proxy fence 也跳过（**不正确**，仅测上界）
  const bool nobp = windowed && (win <= -3);  // 哨兵：连背压/head 也跳过，只发 store（**不正确**，仅测上界）
  const bool skipbp = windowed && (win == -4);  // 哨兵：只跳过背压（保留 head）
  const bool skiphead = windowed && (win == -5);  // 哨兵：只跳过 head（保留背压）

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
    const int NWU = DW >> 2;

    if (mode == 7) {
      // ---------- mode 7: pure-big —— 每 NB 个 batch 合成一次大块 TMA store（测 store 粒度） ----------
      const int NB = (tw < 1) ? 1 : tw;  // 每次 store 的 batch 数（复用 tw 参数）
      uint32_t* stage = smem;
      const uint32_t group_words = (uint32_t)NB * (uint32_t)K * (uint32_t)DW;
      const long long ngroups = nbp / NB;
      for (long long g = 0; g < ngroups; ++g) {
        if (lane == 0 && g >= 2) bulk_wait_read_n(1);
        __syncthreads();
        uint32_t* slot = stage + (size_t)(g % 2) * group_words;
        for (uint32_t i = lane; i < group_words; i += nlanes) {
          uint32_t d = i / (uint32_t)DW;
          uint32_t w = i % (uint32_t)DW;
          uint32_t gid = (uint32_t)(off + g * (long long)NB * K + d);
          slot[i] = gen_word(gid, (int)w);
        }
        __syncthreads();
        if (lane == 0) {
          fence_proxy_async_shared();
          uint32_t* dst =
              ring + (size_t)((size_t)p * CAPp + (size_t)((g * (long long)NB * K) % CAPp)) * DW;
          bulk_store_1d(dst, slot, group_words * 4u);
          bulk_commit();
        }
        __syncthreads();
      }
      if (lane == 0) bulk_wait_all();
      return;
    }

    if (gvar == 3 && !(DW & 3) && (is_gen || smem_only)) {
      // ---------- gvar 3: TMA bulk store（batch 粒度，sb 槽 SMEM 暂存） ----------
      uint32_t* stage = smem;  // sb 个槽，每槽 K*DW 个字
      const uint32_t stage_words = (uint32_t)K * (uint32_t)DW;
      const int eff_win = windowed ? (nowait ? 1 : (win < 1 ? 1 : (win > sb ? sb : win))) : 1;
      for (long long bj = 0; bj < nbp; ++bj) {
        // 背压：等本 producer 已消费到 (bj+1)*K 条（pure 档 CAP 取满、等价无背压）
        if (is_gen && !nobp && !skipbp) {
          if (lane == 0) {
            unsigned long long need = (unsigned long long)((bj + 1) * K);
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
        // 回收槽：等最老的 store 读完源（bj>=sb 时同槽的旧 store 应已读完）
        if (lane == 0 && bj >= sb) bulk_wait_read_n(sb - 1);
        __syncthreads();
        uint32_t* slot = stage + (size_t)(bj % sb) * stage_words;
        // 协作把 K 条描述符生成到 SMEM 槽（内容 = gen_word）
        for (uint32_t i = lane; i < stage_words; i += nlanes) {
          uint32_t d = i / (uint32_t)DW;
          uint32_t w = i % (uint32_t)DW;
          uint32_t gid = (uint32_t)(off + bj * K + d);
          slot[i] = gen_word(gid, (int)w);
        }
        // 让本线程生成的 SMEM 对其它线程可见，再由 lane0 发 async proxy
        __syncthreads();
        if (lane == 0) {
          fence_proxy_async_shared();
          if (!smem_only) {
            uint32_t* dst = ring + (size_t)((size_t)p * CAPp + (size_t)(((bj * K)) % CAPp)) * DW;
            bulk_store_1d(dst, slot, stage_words * 4u);
            bulk_commit();
        if (is_gen && !nobp && !skiphead) {
              if (windowed) {
                if (nowait) {
                  // 上界口径（**不正确**：不等 store 完成就发布 head，跨 proxy 可见性无保证）
                  if (!nofence) fence_proxy_async_global();
                  st_release_u64(&head[p], (unsigned long long)((bj + 1) * K));
                } else if (bj + 1 >= eff_win) {
                  // 流水正确握手：只需保证「要发布的那批」的 store 已完成
                  bulk_wait_n(eff_win - 1);
                  fence_proxy_async_global();
                  st_release_u64(&head[p],
                                 (unsigned long long)((bj - eff_win + 2) * K));
                }
              } else {
                // serialized 握手（§50 的 gen）：每批全 drain
                bulk_wait_all();
                fence_proxy_async_global();
                st_release_u64(&head[p], (unsigned long long)((bj + 1) * K));
              }
            }
          }
        }
        // 给消费侧一点工作量把它藏住（PW）
        unsigned int pv = (unsigned int)((long long)bj * K + lane);
        for (int r = 0; r < PW; ++r) pv = pv * 1664525u + 1013904223u;
        pacc += pv;
        __syncthreads();
      }
      if (!smem_only && lane == 0) {
        bulk_wait_all();
        if (is_gen && windowed) {  // 补发尾部剩余批的 head
          fence_proxy_async_global();
          st_release_u64(&head[p], (unsigned long long)(nbp * K));
        }
      }
      if (lane == 0 && psink && pacc) atomicAdd(psink, pacc);
      return;
    }

    // ---------- gvar 0/1/2: 寄存器 store（沿用 §48/§49 的 per-step 结构） ----------
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
  const int nt = (w < nbatch) ? (nbatch - w + Gw - 1) / Gw : 0;

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
    while (h < need) {
      if (abort && *abort) return false;
      __nanosleep(100);
      h = ld_acquire_u64(&head[own]);
    }
    return true;
  };

  auto issue = [&](int t) -> bool {
    int slot = t % D;
    int bb = w + t * Gw;
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
    int bb = w + t * Gw;
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

// P = producer 块数；G = worker 数（grid = P+G）。CAP = 总环槽数（P 等分）。
// mode 5(pure)/6(sgen)/7(pure-big)/9(pure-pipe) 忽略 G，grid = P。
static Res run_scale(int P, int G, long long NDESC, int DW, int EDW, int K, int D, int R, int PW,
                     long long CAPin, int mode, int reps, int T, int gvar, int df = 1, int tw = 1,
                     int win = 1, int sb = SB) {
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
  if (mode == 7) stage_bytes = (size_t)2 * (size_t)tw * K * DW * 4;  // 大块 store 双缓冲
  if (stage_bytes > smem_bytes) smem_bytes = stage_bytes;
  size_t req = smem_bytes;
  if (req < 1024) req = 1024;
  if (req > 48 * 1024)
    CK(cudaFuncSetAttribute(dptp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req));

  ensure_ctl(P);
  ensure_data(NDESC, EDW);
  ensure_static(NDESC, DW);
  if (mode == 1 || mode == 5 || mode == 6 || mode == 7 || mode == 8 || mode == 9)
    ensure_ring(CAP, DW);

  int grid;
  if (mode == 0) grid = G;
  else if (mode == 5 || mode == 6 || mode == 7 || mode == 9) grid = P;
  else grid = P + G;

  for (int w = 0; w < 2; ++w) {
    CK(cudaMemset(g_head, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_tail, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_abort, 0, 4));
    dptp_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, gvar,
                                  df, tw, win, sb, g_static, g_ring, g_head, g_tail, g_data, g_sink,
                                  g_psink, g_abort);
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
    dptp_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, gvar,
                                  df, tw, win, sb, g_static, g_ring, g_head, g_tail, g_data, g_sink,
                                  g_psink, g_abort);
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
  printf("  P=%-3d G=%-3d EDW=%-5d K=%-4d D=%-3d R=%-6d PW=%-6d CAPp=%-6lld mode=%d gvar=%d win=%d  "
         "%9.4f ms/rep  got=%llu%s\n",
         P, G, EDW, K, D, R, PW, CAPp, mode, gvar, win, res.ms, res.got,
         aborted ? "  ABORT" : "");
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return res;
}

int main(int argc, char** argv) {
  setvbuf(stdout, NULL, _IONBF, 0);
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  const int NSM = prop.multiProcessorCount;
  const size_t OPTIN = prop.sharedMemPerBlockOptin;
  printf("device %d: %s  SMs=%d  cc=%d.%d  L2=%d MB  optin=%zu  maxThreads/SM=%d\n", dev,
         prop.name, NSM, prop.major, prop.minor, (int)(prop.l2CacheSize >> 20), OPTIN,
         prop.maxThreadsPerMultiProcessor);
  CK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

  const int DW = 88;                // 352 B, 对齐 MPK §6.1
  const int T = 256;
  const long long NDESC = 393216;   // ×352 B ≈ 132 MB ≫ L2（静态数组口径）
  const int K = 32, D_ = 2;

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
    int tw = argc > 17 ? atoi(argv[17]) : 1;
    int win = argc > 18 ? atoi(argv[18]) : 1;
    int sb = argc > 19 ? atoi(argv[19]) : SB;
    run_scale(P, G, N, dw, edw, Kk, D, R, PW, CAP, mode, reps, t, gvar, df, tw, win, sb);
    return 0;
  }

  const size_t SMEM = (size_t)D_ * K * DW * 4 + D_ * 8;
  printf("worker/生产者 全共驻：grid = P + G = %d SM。K=%d D=%d SMEM=%zu B  stage(SB=%d)=%zu B\n",
         NSM, K, D_, SMEM, SB, (size_t)SB * K * DW * 4);
  printf("静态数组 NDESC·DW·4 = %.1f MB (≫ L2)\n", (double)NDESC * DW * 4 / 1e6);

  // §A 正确性：serialized（mode1）vs pipelined（mode8, win 2..4），TMA store，D=1，有界子环 + 背压
  printf("\n===== §A 正确性：TMA store serialized(win=1) vs pipelined(win=2..4)，NDESC=8192, EDW=8, R=0 =====\n");
  {
    long long N = 8192;
    unsigned long long ref = 0;
    int G = NSM - 4;
    for (int P : {1, 2}) {
      long long CAP = (long long)P * K * 8;
      long long lead = (long long)1 * NSM * K;
      if (CAP < lead) CAP = lead;
      Res st = run_scale(P, G, N, DW, 8, K, 1, 0, 0, 0, 0, 1, T, 3);  // static 参照
      Res g1 = run_scale(P, G, N, DW, 8, K, 1, 0, 0, CAP, 1, 1, T, 3, 1, 1, 1);
      if (P == 1) ref = st.got;
      printf("      P=%d  static%s  serialized(win=1)%s\n", P, st.got == ref ? "==" : "!=",
             g1.got == ref ? "==" : "!=");
      for (int win : {2, 3, 4}) {
        Res gp = run_scale(P, G, N, DW, 8, K, 1, 0, 0, CAP, 8, 1, T, 3, 1, 1, win);
        printf("      P=%d  pipelined (win=%d)%s\n", P, win, gp.got == ref ? "==" : "!=");
      }
    }
  }

  // §B headline：流水握手把 gen 相对 static 的代价压到多少（有 worker，EDW=0, PW=0）
  printf("\n===== §B TMA store gen 握手 × P（EDW=0, PW=0；grid=P+G=NSM 全共驻；static=参照）=====\n");
  {
    long long CAPbig = NDESC;
    Res s = run_scale(1, NSM, NDESC, DW, 0, K, D_, 0, 0, 0, 0, 7, T, 0);
    double bd = (double)NDESC * DW * 4.0;
    printf("      static(G=%d, 无 producer) = %9.4f ms  desc=%8.1f GB/s\n", NSM, s.ms,
           bd / (s.ms / 1e3) / 1e9);
    printf("  %-4s %-4s %11s %9s %11s %9s %11s %9s %11s %9s %11s %9s %11s %9s\n", "P", "G",
           "ser(ms)", "ser/st", "pipe2(ms)", "p2/st", "pipe3(ms)", "p3/st", "pipe4(ms)", "p4/st",
           "v4(ms)", "v4/st", "co(ms)", "co/st");
    for (int P : {1, 2, 4, 8, 16, 32, 64}) {
      if (P >= NSM) break;
      int G = NSM - P;
      Res gs = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 1, 7, T, 3, 1, 1, 1);
      Res g2 = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 8, 7, T, 3, 1, 1, 2);
      Res g3 = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 8, 7, T, 3, 1, 1, 3);
      Res g4 = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 8, 7, T, 3, 1, 1, 4);
      Res gv = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 1, 7, T, 1);
      Res gc = run_scale(P, G, NDESC, DW, 0, K, D_, 0, 0, CAPbig, 1, 7, T, 2);
      printf("  %-4d %-4d %11.4f %9.3f %11.4f %9.3f %11.4f %9.3f %11.4f %9.3f %11.4f %9.3f "
             "%11.4f %9.3f\n",
             P, G, gs.ms, gs.ms / s.ms, g2.ms, g2.ms / s.ms, g3.ms, g3.ms / s.ms, g4.ms,
             g4.ms / s.ms, gv.ms, gv.ms / s.ms, gc.ms, gc.ms / s.ms);
    }
  }

  // §C 纯写吞吐（mode 5 ser / mode 9 pipe / mode 7 rolling，无 worker）——隔离写入路径
  printf("\n===== §C 纯写吞吐（无 worker 消费）：ser(mode5) / pipe(mode9) / rolling(mode7) × P =====\n");
  {
    double bd = (double)NDESC * DW * 4.0;
    printf("  %-4s %11s %11s %11s %11s %11s %11s %11s\n", "P", "ser(GB/s)", "p2(GB/s)",
           "p3(GB/s)", "p4(GB/s)", "roll(GB/s)", "v4(GB/s)", "co(GB/s)");
    for (int P : {1, 2, 4, 8, 16, 32, 64, 128}) {
      if (P > NSM) break;
      Res w5 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 7, T, 3);
      Res w2 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, 2);
      Res w3 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, 3);
      Res w4 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, 4);
      Res w7 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 7, 7, T, 3, 1, 1);
      Res w1 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 7, T, 1);
      Res wc = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 7, T, 2);
      printf("  %-4d %11.1f %11.1f %11.1f %11.1f %11.1f %11.1f %11.1f\n", P,
             bd / (w5.ms / 1e3) / 1e9, bd / (w2.ms / 1e3) / 1e9, bd / (w3.ms / 1e3) / 1e9,
             bd / (w4.ms / 1e3) / 1e9, bd / (w7.ms / 1e3) / 1e9, bd / (w1.ms / 1e3) / 1e9,
             bd / (wc.ms / 1e3) / 1e9);
    }
  }

  // §E 更深窗口：纯写（无 worker），固定 P=16/128，扫更深的 SMEM 暂存深度 sb 与窗口 win
  printf("\n===== §E 更深窗口 × SMEM 暂存槽数 sb（mode 9 pure-pipe，无 worker）=====\n");
  {
    double bd = (double)NDESC * DW * 4.0;
    printf("  %-4s %-4s %-4s %11s %11s %13s\n", "P", "sb", "win", "ms", "GB/s", "vs_sb4_win2");
    for (int P : {16, 128}) {
      if (P > NSM) break;
      double base = 0;
      int sbs[3] = {4, 8, 16};
      for (int i = 0; i < 3; ++i) {
        int sb = sbs[i];
        int wins[4] = {2, 4, sb / 2, sb};
        int prev = -1;
        for (int j = 0; j < 4; ++j) {
          int win = wins[j];
          if (win < 2) win = 2;
          if (win > sb) win = sb;
          if (win == prev) continue;
          prev = win;
          Res r = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, win, sb);
          double bw = bd / (r.ms / 1e3) / 1e9;
          if (i == 0 && j == 0) base = bw;
          printf("  %-4d %-4d %-4d %11.4f %11.1f %13.3f\n", P, sb, win, r.ms, bw, bw / base);
        }
      }
    }
  }

  // §F 代价归因（**上界/哨兵口径，部分不正确**）：把「完成等待 / fence / 背压 / head 发布」逐个拆掉，
  //    定位 serialized 的 472 到底卡在哪一步。win=-1 跳完成等待、win=-2 再跳 fence、
  //    win=-3 再跳背压+head（只剩 store）、win=-4 只跳背压、win=-5 只跳 head。mode 6=sgen（不 store）、
  //    mode 7=rolling（不等待，对上 §50）。
  printf("\n===== §F 代价归因（哨兵口径）：拆掉完成等待/fence/背压/head 各步 × P =====\n");
  {
    double bd = (double)NDESC * DW * 4.0;
    printf("  %-4s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", "P", "ser", "pipe2", "nowait",
           "nofence", "nobp", "skipbp", "skiphd", "roll(m7)", "sgen(m6)", "v4");
    for (int P : {1, 16, 64, 128}) {
      if (P > NSM) break;
      Res se = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 7, T, 3);
      Res p2 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, 2, 4);
      Res nw = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, -1, 4);
      Res nf = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, -2, 4);
      Res nb = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, -3, 4);
      Res sb = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, -4, 4);
      Res sh = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 9, 7, T, 3, 1, 1, -5, 4);
      Res ro = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 7, 7, T, 3, 1, 1);
      Res sg = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 6, 7, T, 3);
      Res v4 = run_scale(P, 0, NDESC, DW, 0, K, D_, 0, 0, NDESC, 5, 7, T, 1);
      double d = 1e3;
      printf("  %-4d %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", P,
             bd / (se.ms / d) / 1e9, bd / (p2.ms / d) / 1e9, bd / (nw.ms / d) / 1e9,
             bd / (nf.ms / d) / 1e9, bd / (nb.ms / d) / 1e9, bd / (sb.ms / d) / 1e9,
             bd / (sh.ms / d) / 1e9, bd / (ro.ms / d) / 1e9, bd / (sg.ms / d) / 1e9,
             bd / (v4.ms / d) / 1e9);
    }
  }

  // §D 接数据面：P=16 固定，serialized vs pipelined(win=2,3,4) TMA store，扫 EDW
  printf("\n===== §D 接数据面：TMA store ser vs pipe 扫 EDW（P=16；K=%d,D=%d,PW=0）=====\n", K, D_);
  {
    long long CAPbig = NDESC;
    printf("  %-6s %11s %11s %11s %11s %11s %9s %9s\n", "EDW", "ser", "pipe2", "pipe3", "pipe4",
           "exec_only", "ser/max", "p4/max");
    for (int EDW : {0, 16, 64, 256, 1024, 4096}) {
      Res gs = run_scale(16, NSM - 16, NDESC, DW, EDW, K, D_, 0, 0, CAPbig, 1, 5, T, 3, 1, 1, 1);
      Res g2 = run_scale(16, NSM - 16, NDESC, DW, EDW, K, D_, 0, 0, CAPbig, 8, 5, T, 3, 1, 1, 2);
      Res g3 = run_scale(16, NSM - 16, NDESC, DW, EDW, K, D_, 0, 0, CAPbig, 8, 5, T, 3, 1, 1, 3);
      Res g4 = run_scale(16, NSM - 16, NDESC, DW, EDW, K, D_, 0, 0, CAPbig, 8, 5, T, 3, 1, 1, 4);
      Res e = run_scale(1, NSM, NDESC, DW, EDW, K, D_, 0, 0, 0, 0, 5, T, 0, 0);  // df=0 纯执行
      double mx = (gs.ms > e.ms ? gs.ms : e.ms);
      printf("  %-6d %11.4f %11.4f %11.4f %11.4f %11.4f %9.3f %9.3f\n", EDW, gs.ms, g2.ms, g3.ms,
             g4.ms, e.ms, gs.ms / mx, g4.ms / mx);
    }
  }

  if (g_static) cudaFree(g_static);
  if (g_ring) cudaFree(g_ring);
  if (g_data) cudaFree(g_data);
  if (g_head) { cudaFree(g_head); cudaFree(g_tail); }
  cudaFree(g_sink);
  cudaFree(g_psink);
  cudaFree(g_abort);
  return 0;
}
