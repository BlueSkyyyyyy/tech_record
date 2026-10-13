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
                            int sb, int instr, const uint32_t* __restrict__ static_ring,
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
                       mode == 12 || mode == 13);
  const bool smem_only = (mode == 6);
  // ---- 归因矩阵：显式位（win=1 正确 serialized；win<0 的哨兵档按下面映射）----
  //   WA 完成等待(bulk_wait_all)  BP 背压(tail acquire + syncthreads)  HD 发布 head
  //   REL head 用 release(1)/relaxed(0)  ACQ tail 用 acquire(1)/relaxed(0)
  bool WA = true, BP = true, HD = true, REL = true, ACQ = true, FENCE = true;
  int bps = 1, pbs = 1;  // 背压 / 发布的摊薄步长
  const int S = (bpstride < 1) ? 1 : bpstride;
  if (mode == 9) {
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
        // 协作把 K 条描述符生成到 SMEM 槽（内容 = gen_word）
        for (uint32_t i = lane; i < stage_words; i += nlanes) {
          uint32_t d = i / (uint32_t)DW;
          uint32_t w = i % (uint32_t)DW;
          uint32_t gid = (uint32_t)(off + bj * K + d);
          slot[i] = gen_word(gid, (int)w);
        }
        __syncthreads();
        if (instr && lane == 0 && bj > 0) {
          t_gen += clock64() - c0;
          c0 = clock64();
        }
        if (lane == 0) {
          fence_proxy_async_shared();
          if (!smem_only) {
            uint32_t* dst = ring + (size_t)((size_t)p * CAPp + (size_t)(((bj * K)) % CAPp)) * DW;
            bulk_store_1d(dst, slot, stage_words * 4u);
            bulk_commit();
            if (instr && bj > 0) {
              t_store += clock64() - c0;
              c0 = clock64();
            }
            // 发布 head（或跳过 / 摊薄）
            bool pub_now = do_pub && (pbs == 1 || (bj % pbs == pbs - 1) || (bj == nbp - 1));
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
                     int bpstride = 1, int win = 1, int sb = SB, int instr = 0) {
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
  else grid = P + G;

  for (int w = 0; w < 2; ++w) {
    CK(cudaMemset(g_head, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_tail, 0, sizeof(unsigned long long) * P));
    CK(cudaMemset(g_abort, 0, 4));
    dptd_kernel<<<grid, T, req>>>(P, (int)NDESC, DW, EDW, K, D, R, PW, (int)CAP, mode, bp, gvar,
                                  df, bpstride, win, sb, instr, g_static, g_ring, g_head, g_tail,
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
                                  df, bpstride, win, sb, instr, g_static, g_ring, g_head, g_tail,
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
         mode == 12 || mode == 13;
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
    run_scale(P, G, N, dw, edw, Kk, D, R, PW, CAP, mode, reps, t, gvar, df, bpstride, win, sb,
              instr);
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
