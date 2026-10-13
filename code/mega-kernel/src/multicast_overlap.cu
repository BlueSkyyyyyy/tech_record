// ============================================================================
// P171-1（§96）：TMA multicast 的「真正重叠掩码」——同一 CTA 被多笔 multicast 打到
//        **同一 SMEM 槽**时的语义 / 正确性 / 成本。补 §95.7 明留的尾巴：
//        「§95 四种 pattern 下每个 CTA 恒为唯一一次的接收者；真正的『重叠掩码』
//         （多笔 multicast 写同一槽的竞争序）未验证」。
//
// 背景：
//   - §19/§20/§93/§94/§95 的 multicast 掩码无论连续/离散，**每个接收者每轮恰被投递一次**
//     （每 CTA 是唯一组的唯一接收者）。因此「同一 SMEM 槽被两笔异步 TMA 写」从未被测。
//   - 生产 megakernel（DeepGEMM Mega MoE 「Always multicast on A」、`docs/02` §1.5）里，
//     若两个 CTA 各自广播不同的 tile 到重叠接收集，就可能出现「同槽多写」。本节量清它：
//       * payload 相同时是否**幂等**（确定性正确）？
//       * payload 不同时是否出现**竞争序 / 撕裂**（无定义）？
//       * 重叠发送的**成本**是「去重后的接收者数」还是「发送笔数 × 接收者数」？
//
// 载体：§19/§93/§94/§95 `bcast_kernel` / `shape_kernel` 的忠实复刻（SMEM chunk 槽、
//       每轮发 TMA、一个 mbarrier `expect_tx` 累计本次到达的全部字节后一次等待、`cl.sync()`）。
//       唯一变量 = **几个发起者 + 写哪个槽 + 各发起者读哪一行 payload**：
//
//   mode 0  one   ：1 个发起者（rank0），写槽 0，payload 行 r          —— §95 基线
//   mode 1  idem  ：N 个发起者（rank0..N-1），**全写槽 0**，payload 行 **相同**（幂等重叠）
//   mode 2  race  ：N 个发起者（rank0..N-1），**全写槽 0**，payload 行 **互异**（竞争重叠）
//   mode 3  nslot ：N 个发起者，发起者 i 写**槽 i**，payload 行互异（不相交槽、重叠接收者）
//   mode 4  nslot1：同 mode 3 的投递（N 个不同槽），但**每线程只读槽 0**（隔离「投递成本」与
//                   「N 槽读回成本」的诊断变体）
//
// 每笔 multicast 的 `mask` 恒为整群 `(1<<C)-1`（发起者都在自己掩码内）。故：
//   mode0 每轮到达 = 1·chunk；mode1/2 = N·chunk 写同一槽；mode3 = N·chunk 写 N 个槽。
//   每接收者的 `expect_tx` = 实际到达字节（mode0: chunk，其余: N·chunk）。
//
// 待答问题：
//   Q1（§A）：**幂等重叠**（mode1）在任意 N / C 下是否**确定性正确**（收到的恰是那笔
//       payload，与到达顺序无关）？
//   Q2（§A）：**竞争重叠**（mode2）的最终槽内容是否是**某一位发送者的 payload**（无撕裂，
//       但胜者不确定），还是会出现**混合/撕裂**？
//   Q3（§B）：重叠发送的成本是「popcount 之和」还是「去重后接收者数」？即硬件是否对
//       同槽重叠做去重/合并（delivered、源侧 L2 请求、墙钟随 N 的标度）。
//   Q4（§C）：**不相交槽 + 重叠接收者**（mode3）是否确定性正确、成本 = N×基线？
//   Q5（§D，ncu）：idem 模式下 N 个发起者读**同一源行**，源侧 L2 读 sector 是否被合并
//       （还是 N 倍）？
//
// 编译：nvcc -O3 -arch=sm_90a src/multicast_overlap.cu -o /tmp/multicast_overlap -lcuda
// 用法：见 harness/multicast_overlap.sh（每配置独立进程——cluster launch failure 会 sticky）
//   B <C> <grid> <chunkKB> <R> <L> <mode> <N> <iters>   # 计时（+ mode0/1/3 正确性）
//   P <C> <grid> <chunkKB> <L> <mode> <N> <reps>        # 正确性探针（R=1，逐次读回）
//   目标卡 H100 / sm90a。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
  const char* s = "?"; cuGetErrorString(r, &s); \
  printf("[CU FAIL] %s -> %s\n", #x, s); exit(2); } } while (0)
#define CE(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("[CUDA FAIL] %s -> %s\n", #x, cudaGetErrorString(e)); exit(2); } } while (0)

static const int TH = 128;

static CUtensorMap make_tmap_2d(const float* g, uint64_t d0, uint64_t d1,
                                uint32_t b0, uint32_t b1) {
  CUtensorMap tmap;
  uint64_t globalDim[2] = {d0, d1};
  uint64_t globalStrides[1] = {d0 * 4};
  uint32_t boxDim[2] = {b0, b1};
  uint32_t elemStrides[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (void*)g,
      globalDim, globalStrides, boxDim, elemStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { const char* s = "?"; cuGetErrorString(r, &s);
    printf("[tmap encode FAIL] %s\n", s); exit(2); }
  return tmap;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               ::"r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred P;\n"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@!P bra WAIT_%=;\n\t}"
      ::"r"(smem_u32(mbar)), "r"(phase) : "memory");
}
__device__ __forceinline__ void tma_load_mcast_2d(const CUtensorMap* tmap, uint32_t dst,
                                                  uint64_t* mbar, uint16_t mask,
                                                  int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%3, %4}], [%2], %5;"
      ::"r"(dst), "l"((uint64_t)tmap), "r"(smem_u32(mbar)), "r"(c0), "r"(c1), "h"(mask)
      : "memory");
}

// mode 0 one  : 1 发起者, 槽0, payload 行 r
// mode 1 idem : N 发起者, 槽0, payload 行相同 (独占幂等)
// mode 2 race : N 发起者, 槽0, payload 行互异 (竞争)
// mode 3 nslot: N 发起者, 槽 i, payload 行互异
__global__ void overlap_kernel(const __grid_constant__ CUtensorMap tmap,
                               float* __restrict__ out,
                               int R, int box1, int L, int mode, int N) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  float* buf = reinterpret_cast<float*>(smem_raw);
  __shared__ __align__(8) uint64_t mbar;

  cg::cluster_group cl = cg::this_cluster();
  int C = cl.num_blocks();
  int rank = cl.block_rank();
  int chunk = 256 * box1;
  uint32_t chunk_bytes = (uint32_t)chunk * 4u;
  uint16_t mask = (C >= 16) ? 0xFFFFu : (uint16_t)((1u << C) - 1u);

  if (threadIdx.x == 0) { mbar_init(&mbar, 1); mbar_init_fence(); }
  cl.sync();

  float acc = 0.f;
  uint32_t phase = 0;
  bool readall = (mode == 3);
  for (int r = 0; r < R; r++) {
    if (threadIdx.x == 0) {
      uint32_t expect = (mode == 0) ? chunk_bytes : (uint32_t)N * chunk_bytes;
      mbar_expect_tx(&mbar, expect);
      for (int i = 0; i < N; i++) {
        if (rank == i) {
          int lane = (mode == 1) ? 0 : i;          // idem: 所有发起者读同一行
          int col = (r * N + lane) * box1;
          int slot = (mode == 3 || mode == 4) ? i : 0;
          uint32_t dst = smem_u32(buf + (size_t)slot * chunk);
          tma_load_mcast_2d(&tmap, dst, &mbar, mask, 0, col);
        }
      }
    }
    mbar_wait(&mbar, phase);
    phase ^= 1u;
    if (readall) {
      for (int i = 0; i < N; i++) {
        acc += buf[(size_t)i * chunk + threadIdx.x];
        acc += buf[(size_t)i * chunk + threadIdx.x + TH];
      }
    } else {
      acc += buf[threadIdx.x];
      acc += buf[threadIdx.x + TH];
    }
    cl.sync();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

static cudaError_t launch_cluster(int C, dim3 grid, dim3 block, size_t smem,
                                  void* kern, void** args) {
  cudaFuncSetAttribute((const void*)kern,
                       cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (smem >= 48 * 1024)
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = 0;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  cfg.attrs = at; cfg.numAttrs = 1;
  return cudaLaunchKernelExC(&cfg, kern, args);
}

static int num_sm() {
  int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
  return n;
}

static const char* mode_name(int m) {
  switch (m) { case 0: return "one"; case 1: return "idem";
               case 2: return "race"; case 3: return "nslot"; case 4: return "nslot1";
               default: return "?"; }
}

static double val_at(long flat) { return (double)((flat % 251) - 125) * 1e-3; }

// 每个 CTA 的期望 acc（确定性模式）；mode2 返回 -999（无定义）
static double expected_acc(int mode, int N, int box1, int R, int gcnt, int tid) {
  if (mode == 2) return -999.0;
  double e = 0;
  for (int r = 0; r < R; r++) {
    if (mode == 0) {
      long col = (long)r * box1;
      e += val_at(col * 256 + tid) + val_at(col * 256 + tid + TH);
    } else if (mode == 1) {
      long col = (long)r * N * box1;              // 所有发起者读到同一行
      e += val_at(col * 256 + tid) + val_at(col * 256 + tid + TH);
    } else if (mode == 3) {
      for (int i = 0; i < N; i++) {
        long col = ((long)r * N + i) * box1;
        e += val_at(col * 256 + tid) + val_at(col * 256 + tid + TH);
      }
    } else if (mode == 4) {                       // nslot1: 只读槽 0（lane 0）
      long col = (long)r * N * box1;
      e += val_at(col * 256 + tid) + val_at(col * 256 + tid + TH);
    } else { (void)gcnt; }
  }
  return e;
}

// mode2 / mode3 每轮的候选 payload 行（每 CTA 的候选 acc）
static void race_candidates(int N, int box1, int r, int tid, double* cand, int* ncand) {
  for (int i = 0; i < N; i++) {
    long col = ((long)r * N + i) * box1;
    cand[i] = val_at(col * 256 + tid) + val_at(col * 256 + tid + TH);
  }
  *ncand = N;
}

static double* alloc_src(long ntot) {
  float* src;
  CE(cudaMalloc(&src, ntot * 4));
  float* h = (float*)malloc(ntot * 4);
  for (long i = 0; i < ntot; i++) h[i] = (float)((i % 251) - 125) * 1e-3f;
  CE(cudaMemcpy(src, h, ntot * 4, cudaMemcpyHostToDevice));
  free(h);
  return (double*)(void*)src;
}

// -------- 计时命令 B --------
static double run_overlap(int C, int grid, int KB, int R, int L, int mode, int N, int iters,
                          double* maxerr, int* occ_out, int* regs_out) {
  int box1 = KB * 1024 / 4 / 256;
  int chunk = 256 * box1;
  if (mode < 0 || mode > 4) { printf("[bad mode] "); return -1; }
  if (mode == 0 && N != 1) { printf("[mode0 needs N=1] "); return -1; }
  if (mode != 0 && N < 2) { printf("[overlap needs N>=2] "); return -1; }
  if (N > C) { printf("[N>C] "); return -1; }
  if ((mode == 3 || mode == 4) && L < N) { printf("[nslot needs L>=N] "); return -1; }
  long ntot = 256L * (long)box1 * (long)R * (long)N;
  float* src = (float*)alloc_src(ntot);
  float* out;
  CE(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  CUtensorMap tmap = make_tmap_2d(src, 256, ntot / 256, 256, box1);

  size_t smem = (size_t)L * chunk * 4;
  void* kern = (void*)overlap_kernel;
  if (smem >= 48 * 1024)
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  int occ = 0, regs = 0;
  {
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, (const void*)kern); regs = fa.numRegs;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, (const void*)kern, TH, smem);
  }
  void* args[] = {(void*)&tmap, (void*)&out, (void*)&R, (void*)&box1, (void*)&L,
                  (void*)&mode, (void*)&N};
  cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, kern, args);
  if (e != cudaSuccess) { printf("[FAIL %s] ", cudaGetErrorString(e)); return -1; }
  CE(cudaDeviceSynchronize());

  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; i++) {
    e = launch_cluster(C, dim3(grid), dim3(TH), smem, kern, args);
    if (e != cudaSuccess) { printf("[iter FAIL %s] ", cudaGetErrorString(e)); return -1; }
  }
  CE(cudaDeviceSynchronize());
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;

  // 正确性（确定性模式；mode2 竞争无定义，交 P 探针）
  double err = -1;
  if (mode != 2) {
    float* o = (float*)malloc((size_t)grid * sizeof(float));
    CE(cudaMemcpy(o, out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost));
    err = 0;
    for (int gi = 0; gi < grid; gi++) {
      double exp = expected_acc(mode, N, box1, R, C, 0);
      double d = o[gi] - exp; if (d < 0) d = -d;
      if (d > err) err = d;
    }
    free(o);
  }
  *maxerr = err;
  *occ_out = occ; *regs_out = regs;
  cudaFree(src); cudaFree(out);
  return ms;
}

// -------- 竞争探针 P（R=1，逐次读回） --------
static int probe_race(int C, int grid, int KB, int L, int mode, int N, int reps) {
  int box1 = KB * 1024 / 4 / 256;
  int chunk = 256 * box1;
  int R = 1;
  long ntot = 256L * (long)box1 * (long)R * (long)N;
  if (ntot < 4 * 256) ntot = 4 * 256;
  float* src = (float*)alloc_src(ntot);
  float* out;
  CE(cudaMalloc(&out, (size_t)grid * sizeof(float)));
  CUtensorMap tmap = make_tmap_2d(src, 256, ntot / 256, 256, box1);
  size_t smem = (size_t)L * chunk * 4;
  void* kern = (void*)overlap_kernel;
  if (smem >= 48 * 1024)
    cudaFuncSetAttribute((const void*)kern,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  void* args[] = {(void*)&tmap, (void*)&out, (void*)&R, (void*)&box1, (void*)&L,
                  (void*)&mode, (void*)&N};

  int winners[64]; for (int i = 0; i < 64; i++) winners[i] = 0;
  int torn = 0, total = 0;
  double maxnearest = 0;
  double* cand = (double*)malloc(sizeof(double) * N);
  int ncand = 0;
  float* o = (float*)malloc((size_t)grid * sizeof(float));
  for (int rep = 0; rep < reps; rep++) {
    cudaError_t e = launch_cluster(C, dim3(grid), dim3(TH), smem, kern, args);
    if (e != cudaSuccess) { printf("[P FAIL %s]\n", cudaGetErrorString(e)); return -1; }
    CE(cudaDeviceSynchronize());
    CE(cudaMemcpy(o, out, (size_t)grid * sizeof(float), cudaMemcpyDeviceToHost));
    if (mode == 3) {                              // nslot：读 N 槽，确定性单候选
      double e = 0;
      for (int i = 0; i < N; i++) {
        long col = (long)i * box1;                // r=0, lane i
        e += val_at(col * 256 + 0) + val_at(col * 256 + TH);
      }
      cand[0] = e; ncand = 1;
    } else {                                      // one/idem/nslot1：只读槽0；race：N 候选
      race_candidates(N, box1, 0, 0, cand, &ncand);
    }
    for (int gi = 0; gi < grid; gi++) {
      double best = 1e30; int bi = -1;
      for (int i = 0; i < ncand; i++) {
        double d = o[gi] - cand[i]; if (d < 0) d = -d;
        if (d < best) { best = d; bi = i; }
      }
      total++;
      if (best > 1e-3) { torn++; if (best > maxnearest) maxnearest = best; }
      else winners[bi]++;
    }
  }
  free(cand); free(o); cudaFree(src); cudaFree(out);
  printf("P C=%2d grid=%4d chunk=%2dKB L=%2d mode=%d(%s) N=%2d R=1 reps=%d | torn=%d/%d "
         "max_nearest=%.4f | winner_hist=[", C, grid, KB, L, mode, mode_name(mode), N, reps,
         torn, total, maxnearest);
  for (int i = 0; i < (ncand > 0 ? ncand : N); i++) printf("%d%s", winners[i], i + 1 < (ncand > 0 ? ncand : N) ? "," : "");
  printf("]\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("usage: multicast_overlap B <C> <grid> <chunkKB> <R> <L> <mode> <N> <iters>\n");
    printf("       multicast_overlap P <C> <grid> <chunkKB> <L> <mode> <N> <reps>\n");
    return 1;
  }
  CK(cuInit(0));
  const char* cmd = argv[1];

  if (!strcmp(cmd, "B")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]), R = atoi(argv[5]),
        L = atoi(argv[6]), mode = atoi(argv[7]), N = atoi(argv[8]),
        iters = argc > 9 ? atoi(argv[9]) : 20;
    double err = -1; int occ = 0, regs = 0;
    double ms = run_overlap(C, grid, KB, R, L, mode, N, iters, &err, &occ, &regs);
    int box1 = KB * 1024 / 4 / 256;
    double chunk = 256.0 * box1 * 4;
    double deliv = (double)grid * chunk * (double)N * R;
    double issued = (double)(grid / C) * (double)N * chunk * R;   // 源侧理论读取
    int nsm = num_sm();
    int eff = occ > 0 ? occ : 1;
    int per_sm = grid / nsm;
    if (per_sm < eff) eff = per_sm;
    if (eff < 1) eff = 1;
    double waves = (double)grid / (double)(nsm * eff);
    printf("B C=%2d grid=%4d chunk=%2dKB L=%2d R=%4d mode=%d(%s) N=%2d it=%2d | regs=%3d occ=%2d "
           "eff=%2d waves=%4.2f | ", C, grid, KB, L, R, mode, mode_name(mode), N, iters,
           regs, occ, eff, waves);
    if (ms >= 0)
      printf("%8.3f us | deliver %7.1f GB/s | per-blk %6.1f GB/s | src-issued %7.1f GB/s | maxerr %.6f\n",
             ms * 1e3, deliv / (ms / 1e3) / 1e9, deliv / (ms / 1e3) / 1e9 / grid,
             issued / (ms / 1e3) / 1e9, err);
    else printf("FAILED\n");
  } else if (!strcmp(cmd, "P")) {
    int C = atoi(argv[2]), grid = atoi(argv[3]), KB = atoi(argv[4]),
        L = atoi(argv[5]), mode = atoi(argv[6]), N = atoi(argv[7]),
        reps = argc > 8 ? atoi(argv[8]) : 8;
    probe_race(C, grid, KB, L, mode, N, reps);
  } else { printf("unknown cmd %s\n", cmd); return 1; }
  return 0;
}
