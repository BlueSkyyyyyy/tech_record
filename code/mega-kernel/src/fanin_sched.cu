// fanin_sched.cu — MegaKernel 专题 P22-1 小实验：
//   **去中心化调度的 DAG 形状敏感性：扇入/扇出 × 静态分配 vs 动态队列**
//
// 目的（目标卡 H100 / sm90 实测）：
//   P15-1 用「C 条链」把 MPK §5 的去中心化调度（C5）跑了起来，但那是最退化的 DAG——每个 task
//   恰好 1 个前驱、1 个后继。P15-1 §9.6 明确留了尾巴：「未测 task 数远超队列容量、多级依赖扇入
//   扇出（当前每 task 恰好 1 个后继）」；P20-1 §14 又得到结论「同步必须批量/环形化，而非每 tile
//   原子」。本节把 DAG 推广成**分层扇入/扇出图**（L 层 × W 宽 × 每 task FIN 个前驱），在同一 DAG、
//   同一总工作量上公平对照四种设备内协调机制：
//
//   barrier : 单 cooperative persistent kernel，逐层 grid.sync()（集中式，P5-2/P13-1 形态）
//   wave    : 单 persistent kernel，**无 grid.sync**——每层一个原子工作计数器（worker atomicAdd 抢
//             task）+ 一层完成计数器做「软屏障」，支持 **批量取 task**（一个 atomicAdd 领 K 个 task）。
//             这是「层粒依赖 + 细粒分配」的去中心化中间形态（对应 MPK §5.2 批量入队 / §5.3 预取）。
//   decen   : 单 persistent kernel，**逐 task 事件计数器 + 设备侧原子 ready 队列**、worker 自取
//             （完全动态、task 粒依赖；= P15-1 的机制，此处加上了扇入/扇出）。
//   static  : 单 persistent kernel，**静态 grid-stride 分配**（block b 固定处理 t=b,b+G,…），
//             不设中央队列——每个 task 在执行前轮询其前驱的 done 标志（DeepGEMM 式静态依赖路径的微缩）。
//
// 任务图（真实依赖，非合成屏障）：
//   t = l*W + j（层 l∈[0,L)，列 j∈[0,W)）。t 的前驱 = 层 l-1 的 { (j*FIN+k) mod W : k<FIN }；
//   层 0 无前驱。t 的数值 = seg_sum(t 的 SEG 个输入) + Σ_{p∈pred(t)} val[p]。末层写 out。
//   同一求和顺序 → 四种机制与串行参考**逐位一致**（正确性门限）。
//
// 三个 Section：
//   A 扫宽度 W（固定 L/FIN/SEG）：并行度与调度开销随 task 数缩放。
//   B 扫扇入 FIN（固定 L/W）：每 task 完成要原子递减 FIN 个后继计数器，看原子压力随扇出缩放。
//   C 扫 wave 的批量 K（固定 L/W/FIN）：批量取 task 能否把 per-task 调度原子摊薄。
//   D 扫 task 长度 SEG（固定 L/W/FIN）：调度开销是「每 task 固定」的，task 越长越被摊薄。
//   E 扫层数 L（固定 W/FIN/SEG，薄层）：barrier 每层 +1 次 grid.sync 的地板随 L 累积，
//     看「多而薄的层」何时让集中式 barrier 输给去中心化/静态调度（megakernel 的真实区间）。
//
// 编译：nvcc -O3 -arch=sm_90 fanin_sched.cu -o fanin_sched
// 用法：./fanin_sched [iters] [G] [dev]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
             cudaGetErrorString(_e));                                           \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static const int T = 256;  // 每 block 线程数

// 段归约：block 内 grid-stride 求和 + warp/shared 归约，结果广播给 block 所有线程。
__device__ __forceinline__ float seg_sum(const float* __restrict__ in,
                                         long long b, long long e) {
  float s = 0.f;
  for (long long i = b + threadIdx.x; i < e; i += blockDim.x) s += in[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ float sm[T / 32];
  __shared__ float bres;
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) sm[warp] = s;
  __syncthreads();
  if (warp == 0) {
    s = (threadIdx.x < T / 32) ? sm[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = T / 64; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (threadIdx.x == 0) bres = s;
  }
  __syncthreads();
  return bres;
}

// 计算一个 task 的数值：本地段和 + 前驱 val 之和（前驱顺序固定，保证与参考逐位一致）。
__device__ __forceinline__ float task_val(const float* __restrict__ in, int tt, int SEG,
                                          float* __restrict__ val,
                                          const int* __restrict__ pred_off,
                                          const int* __restrict__ pred_list) {
  float s = seg_sum(in, (long long)tt * SEG, (long long)(tt + 1) * SEG);
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = s;
    // 用 __ldcg（bypass L1）读前驱：跨 SM 的写只保证到 L2，普通 load 可能命中旧 L1 行。
    for (int k = pred_off[tt]; k < pred_off[tt + 1]; ++k) v += __ldcg(&val[pred_list[k]]);
    __stcg(&val[tt], v);
  }
  __syncthreads();
  return val[tt];
}

// ---------------- barrier：单 cooperative persistent kernel，逐层 grid.sync() ----------
__global__ void barrier_kernel(const float* __restrict__ in, float* __restrict__ out,
                               float* __restrict__ val,
                               const int* __restrict__ pred_off,
                               const int* __restrict__ pred_list,
                               int L, int W, int SEG) {
  for (int l = 0; l < L; ++l) {
    for (int j = blockIdx.x; j < W; j += gridDim.x) {
      int tt = l * W + j;
      float v = task_val(in, tt, SEG, val, pred_off, pred_list);
      if (threadIdx.x == 0 && l == L - 1) out[j] = v;
    }
    cooperative_groups::this_grid().sync();
  }
}

// ---------------- wave：无 grid.sync；层工作计数器 + 层完成软屏障；支持批量领 task ----------
__global__ void wave_kernel(const float* __restrict__ in, float* __restrict__ out,
                            float* __restrict__ val,
                            const int* __restrict__ pred_off,
                            const int* __restrict__ pred_list,
                            int* __restrict__ work_cnt, int* __restrict__ layer_done,
                            int L, int W, int SEG, int batch) {
  __shared__ int sidx;
  for (int l = 0; l < L; ++l) {
    while (true) {
      if (threadIdx.x == 0) sidx = atomicAdd(&work_cnt[l], batch);
      __syncthreads();
      int idx = sidx;
      if (idx >= W) break;
      int end = idx + batch; if (end > W) end = W;
      for (int j = idx; j < end; ++j) {
        int tt = l * W + j;
        float v = task_val(in, tt, SEG, val, pred_off, pred_list);
        if (threadIdx.x == 0) {
          __threadfence();
          if (l == L - 1) out[j] = v;
          atomicAdd(&layer_done[l], 1);
        }
        __syncthreads();
      }
    }
    // 软屏障：本层所有 task 完成后才允许进入下一层（保证前驱 val 已可见）。
    if (threadIdx.x == 0)
      while (atomicAdd((int*)&layer_done[l], 0) < W) __nanosleep(64);
    __syncthreads();
  }
}

// ---------------- decen：逐 task 事件计数器 + 设备侧原子 ready 队列（P15-1 机制 + 扇入扇出） ---
__device__ __forceinline__ void q_enqueue(int* __restrict__ q_buf,
                                          int* __restrict__ q_ready,
                                          int* __restrict__ q_tail, int task) {
  int idx = atomicAdd(q_tail, 1);
  q_buf[idx] = task;
  __threadfence();
  atomicExch(&q_ready[idx], 1);
}
__device__ __forceinline__ bool q_dequeue(const int* __restrict__ q_buf,
                                          const int* __restrict__ q_ready,
                                          int* __restrict__ q_head,
                                          const int* __restrict__ q_tail,
                                          const int* __restrict__ done_count,
                                          int total, int* out) {
  long long spins = 0;
  while (true) {
    int h = *((volatile int*)q_head);
    int t = *((volatile int*)q_tail);
    if (h < t) {
      if (atomicCAS(q_head, h, h + 1) == h) {
        while (atomicAdd((int*)&q_ready[h], 0) == 0) __nanosleep(64);
        __threadfence();
        *out = q_buf[h];
        return true;
      }
      continue;
    }
    if (atomicAdd((int*)done_count, 0) >= total) return false;
    if (++spins > 2000000000LL) { *out = -2; return false; }
    __nanosleep(128);
  }
}
__global__ void decen_kernel(const float* __restrict__ in, float* __restrict__ out,
                             float* __restrict__ val, int* __restrict__ dep_count,
                             int* __restrict__ q_buf, int* __restrict__ q_ready,
                             int* __restrict__ q_head, int* __restrict__ q_tail,
                             int* __restrict__ done_count,
                             const int* __restrict__ pred_off,
                             const int* __restrict__ pred_list,
                             const int* __restrict__ succ_off,
                             const int* __restrict__ succ_list,
                             int L, int W, int SEG, int total) {
  __shared__ int stask;
  while (true) {
    if (threadIdx.x == 0) {
      int t;
      if (q_dequeue(q_buf, q_ready, q_head, q_tail, done_count, total, &t)) stask = t;
      else stask = -1;
    }
    __syncthreads();
    if (stask < 0) break;
    int tt = stask, l = tt / W, j = tt % W;
    float v = task_val(in, tt, SEG, val, pred_off, pred_list);
    if (threadIdx.x == 0) {
      __threadfence();
      if (l == L - 1) out[j] = v;
      for (int k = succ_off[tt]; k < succ_off[tt + 1]; ++k)
        if (atomicSub(&dep_count[succ_list[k]], 1) == 1)
          q_enqueue(q_buf, q_ready, q_tail, succ_list[k]);
      atomicAdd(done_count, 1);
    }
    __syncthreads();
  }
}

// ---------------- static：静态 grid-stride 分配 + 前驱事件计数器轮询（无中央队列） ----------
__global__ void static_kernel(const float* __restrict__ in, float* __restrict__ out,
                              float* __restrict__ val,
                              const int* __restrict__ pred_off,
                              const int* __restrict__ pred_list,
                              int* __restrict__ dep_count,
                              const int* __restrict__ succ_off,
                              const int* __restrict__ succ_list,
                              int L, int W, int SEG, int total) {
  for (int tt = blockIdx.x; tt < total; tt += gridDim.x) {
    // 等所有前驱完成（前驱总在更低的层、更小的 tt，无环 → 不会自锁）。
    if (threadIdx.x == 0) {
      while (atomicAdd(&dep_count[tt], 0) != 0) __nanosleep(64);
      __threadfence();  // acquire：观察到 dep 归零后再读 val[pred]
    }
    __syncthreads();
    int l = tt / W, j = tt % W;
    float v = task_val(in, tt, SEG, val, pred_off, pred_list);
    if (threadIdx.x == 0) {
      __threadfence();
      if (l == L - 1) out[j] = v;
      for (int k = succ_off[tt]; k < succ_off[tt + 1]; ++k) atomicSub(&dep_count[succ_list[k]], 1);
    }
    __syncthreads();
  }
}

// ---------------- 主机侧：构图 + 参考 ----------------
struct Dag { int L, W, FIN, SEG, total; };

static void build_dag(const Dag& d, int* pred_off, int* pred_list,
                      int* succ_off, int* succ_list, int* dep_count) {
  // 前驱 CSR：层 0 无前驱；层 l>0 的 (l,j) 前驱 = 层 l-1 的 { (j*FIN+k) mod W }。
  for (int tt = 0; tt <= d.total; ++tt) pred_off[tt] = 0;
  int e = 0;
  for (int l = 0; l < d.L; ++l)
    for (int j = 0; j < d.W; ++j) {
      int tt = l * d.W + j;
      pred_off[tt] = e;
      if (l > 0)
        for (int k = 0; k < d.FIN; ++k) {
          int p = (j * d.FIN + k) % d.W;
          pred_list[e++] = (l - 1) * d.W + p;
        }
    }
  pred_off[d.total] = e;
  for (int tt = 0; tt < d.total; ++tt) dep_count[tt] = pred_off[tt + 1] - pred_off[tt];
  // 后继 CSR（由前驱反推）。
  for (int tt = 0; tt <= d.total; ++tt) succ_off[tt] = 0;
  for (int i = 0; i < e; ++i) succ_off[pred_list[i] + 1]++;
  for (int tt = 1; tt <= d.total; ++tt) succ_off[tt] += succ_off[tt - 1];
  int* cur = (int*)malloc(sizeof(int) * (d.total + 1));
  for (int tt = 0; tt <= d.total; ++tt) cur[tt] = succ_off[tt];
  for (int tt = 0; tt < d.total; ++tt)
    for (int k = pred_off[tt]; k < pred_off[tt + 1]; ++k)
      succ_list[cur[pred_list[k]]++] = tt;
  free(cur);
}

static void reference(const float* h_in, const Dag& d, const int* pred_off,
                      const int* pred_list, float* ref) {
  for (int l = 0; l < d.L; ++l)
    for (int j = 0; j < d.W; ++j) {
      int tt = l * d.W + j;
      double s = 0.0;
      for (int i = 0; i < d.SEG; ++i) s += h_in[(long long)tt * d.SEG + i];
      for (int k = pred_off[tt]; k < pred_off[tt + 1]; ++k) s += ref[pred_list[k]];
      ref[tt] = (float)s;
    }
}

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : 20;
  const int G = (argc > 2) ? atoi(argv[2]) : 132;
  const int dev = (argc > 3) ? atoi(argv[3]) : 3;
  setvbuf(stdout, NULL, _IONBF, 0);
  CUDA_CHECK(cudaSetDevice(dev));
  int numSM = 0, coop = 0;
  cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
  cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
  printf("== fanin_sched (目标卡 H100/sm90) ==\n");
  printf("device=%d  SMs=%d  cooperativeLaunch=%d  iters=%d  T=%d  G=%d\n\n",
         dev, numSM, coop, iters, T, G);

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  struct Case { int L, W, FIN, SEG; const char* name; };
  Case cases[] = {
      {16, 256,  2, 4096, "A: L=16 W=256  FIN=2"},
      {16, 512,  2, 4096, "A: L=16 W=512  FIN=2"},
      {16, 1024, 2, 4096, "A: L=16 W=1024 FIN=2"},
      {16, 2048, 2, 4096, "A: L=16 W=2048 FIN=2"},
      {16, 1024, 1, 4096, "B: L=16 W=1024 FIN=1"},
      {16, 1024, 4, 4096, "B: L=16 W=1024 FIN=4"},
      {16, 1024, 8, 4096, "B: L=16 W=1024 FIN=8"},
      {16, 512,  2, 16384, "D: L=16 W=512 FIN=2 SEG=16384"},
      {16, 512,  2, 32768, "D: L=16 W=512 FIN=2 SEG=32768"},
      {32, 256,  2, 8192, "E: L=32  W=256 FIN=2 SEG=8192"},
      {64, 256,  2, 8192, "E: L=64  W=256 FIN=2 SEG=8192"},
      {128, 256, 2, 8192, "E: L=128 W=256 FIN=2 SEG=8192"},
  };
  const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

  for (int ci = 0; ci < ncases; ++ci) {
    Dag d{cases[ci].L, cases[ci].W, cases[ci].FIN, cases[ci].SEG, 0};
    d.total = d.L * d.W;
    if (d.FIN > d.W) { printf("skip (FIN>W)\n"); continue; }

    int* pred_off = (int*)malloc(sizeof(int) * (d.total + 1));
    int* pred_list = (int*)malloc(sizeof(int) * (size_t)d.total * d.FIN);
    int* succ_off = (int*)malloc(sizeof(int) * (d.total + 1));
    int* succ_list = (int*)malloc(sizeof(int) * (size_t)d.total * d.FIN);
    int* dep_count = (int*)malloc(sizeof(int) * d.total);
    build_dag(d, pred_off, pred_list, succ_off, succ_list, dep_count);

    long long ntot = (long long)d.total * d.SEG;
    float* h_in = (float*)malloc(sizeof(float) * ntot);
    for (long long i = 0; i < ntot; ++i) h_in[i] = (float)(i % 1000) * 1e-3f;
    float* ref = (float*)malloc(sizeof(float) * d.total);
    reference(h_in, d, pred_off, pred_list, ref);

    float *d_in, *d_out, *d_val;
    int *d_pred_off, *d_pred_list, *d_succ_off, *d_succ_list, *d_dep;
    int *d_qbuf, *d_qready, *d_qhead, *d_qtail, *d_done, *d_work, *d_ldone;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * ntot));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * d.W));
    CUDA_CHECK(cudaMalloc(&d_val, sizeof(float) * d.total));
    CUDA_CHECK(cudaMalloc(&d_pred_off, sizeof(int) * (d.total + 1)));
    CUDA_CHECK(cudaMalloc(&d_pred_list, sizeof(int) * (size_t)d.total * d.FIN));
    CUDA_CHECK(cudaMalloc(&d_succ_off, sizeof(int) * (d.total + 1)));
    CUDA_CHECK(cudaMalloc(&d_succ_list, sizeof(int) * (size_t)d.total * d.FIN));
    CUDA_CHECK(cudaMalloc(&d_dep, sizeof(int) * d.total));
    CUDA_CHECK(cudaMalloc(&d_qbuf, sizeof(int) * d.total));
    CUDA_CHECK(cudaMalloc(&d_qready, sizeof(int) * d.total));
    CUDA_CHECK(cudaMalloc(&d_qhead, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_qtail, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_done, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(int) * d.L));
    CUDA_CHECK(cudaMalloc(&d_ldone, sizeof(int) * d.L));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * ntot, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pred_off, pred_off, sizeof(int) * (d.total + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pred_list, pred_list, sizeof(int) * (size_t)d.total * d.FIN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_succ_off, succ_off, sizeof(int) * (d.total + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_succ_list, succ_list, sizeof(int) * (size_t)d.total * d.FIN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dep, dep_count, sizeof(int) * d.total, cudaMemcpyHostToDevice));

    // 初始 ready 队列：层 0 的 W 个 task（id = 0..W-1，恰等于槽位）。
    int* h_qready = (int*)calloc(d.total, sizeof(int));
    for (int j = 0; j < d.W; ++j) h_qready[j] = 1;
    int* h_qbuf = (int*)malloc(sizeof(int) * d.total);
    for (int s = 0; s < d.total; ++s) h_qbuf[s] = s;

    auto reset_all = [&]() {
      cudaMemcpyAsync(d_dep, dep_count, sizeof(int) * d.total, cudaMemcpyHostToDevice, 0);
      cudaMemcpyAsync(d_qbuf, h_qbuf, sizeof(int) * d.W, cudaMemcpyHostToDevice, 0);
      cudaMemsetAsync(d_qready, 0, sizeof(int) * d.total, 0);
      cudaMemcpyAsync(d_qready, h_qready, sizeof(int) * d.W, cudaMemcpyHostToDevice, 0);
      cudaMemsetAsync(d_qhead, 0, sizeof(int), 0);
      cudaMemsetAsync(d_done, 0, sizeof(int), 0);
      int val = d.W;
      cudaMemcpyAsync(d_qtail, &val, sizeof(int), cudaMemcpyHostToDevice, 0);
      cudaMemsetAsync(d_work, 0, sizeof(int) * d.L, 0);
      cudaMemsetAsync(d_ldone, 0, sizeof(int) * d.L, 0);
      cudaMemsetAsync(d_val, 0, sizeof(float) * d.total, 0);
      cudaMemsetAsync(d_out, 0, sizeof(float) * d.W, 0);
    };
    auto reset_out = [&]() {
      cudaMemsetAsync(d_val, 0, sizeof(float) * d.total, 0);
      cudaMemsetAsync(d_out, 0, sizeof(float) * d.W, 0);
    };

    float t_b = 0, t_w = 0, t_d = 0, t_s = 0;
    float md_b = 0, md_w = 0, md_d = 0, md_s = 0;
    int fail_b = 0, fail_w = 0, fail_d = 0, fail_s = 0;
    float *h_b = (float*)malloc(sizeof(float) * d.W);
    float *h_w = (float*)malloc(sizeof(float) * d.W);
    float *h_d = (float*)malloc(sizeof(float) * d.W);
    float *h_s = (float*)malloc(sizeof(float) * d.W);
    auto check = [&](float* h, float* md, int* fail) {
      for (int j = 0; j < d.W; ++j) {
        float dd = fabsf(h[j] - ref[(d.L - 1) * d.W + j]) /
                   fmaxf(1.f, fabsf(ref[(d.L - 1) * d.W + j]));
        *md = fmaxf(*md, dd);
        if (dd > 1e-3f) (*fail)++;
      }
    };

    // --- barrier ---
    {
      reset_out();
      void* args[] = {&d_in, &d_out, &d_val, &d_pred_off, &d_pred_list, &d.L, &d.W, &d.SEG};
      CUDA_CHECK(cudaEventRecord(e0, 0));
      for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)barrier_kernel, dim3(G), dim3(T), args, 0, 0));
        if (it < iters - 1) reset_out();
      }
      CUDA_CHECK(cudaEventRecord(e1, 0));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_b, e0, e1));
      CUDA_CHECK(cudaMemcpy(h_b, d_out, sizeof(float) * d.W, cudaMemcpyDeviceToHost));
      check(h_b, &md_b, &fail_b);
    }
    // --- wave (batch=1) ---
    {
      reset_all();
      CUDA_CHECK(cudaEventRecord(e0, 0));
      for (int it = 0; it < iters; ++it) {
        wave_kernel<<<G, T, 0, 0>>>(d_in, d_out, d_val, d_pred_off, d_pred_list,
                                    d_work, d_ldone, d.L, d.W, d.SEG, 1);
        if (it < iters - 1) reset_all();
      }
      CUDA_CHECK(cudaEventRecord(e1, 0));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_w, e0, e1));
      CUDA_CHECK(cudaMemcpy(h_w, d_out, sizeof(float) * d.W, cudaMemcpyDeviceToHost));
      check(h_w, &md_w, &fail_w);
    }
    // --- decen ---
    {
      reset_all();
      CUDA_CHECK(cudaEventRecord(e0, 0));
      for (int it = 0; it < iters; ++it) {
        decen_kernel<<<G, T, 0, 0>>>(d_in, d_out, d_val, d_dep, d_qbuf, d_qready, d_qhead,
                                     d_qtail, d_done, d_pred_off, d_pred_list, d_succ_off,
                                     d_succ_list, d.L, d.W, d.SEG, d.total);
        if (it < iters - 1) reset_all();
      }
      CUDA_CHECK(cudaEventRecord(e1, 0));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_d, e0, e1));
      CUDA_CHECK(cudaMemcpy(h_d, d_out, sizeof(float) * d.W, cudaMemcpyDeviceToHost));
      check(h_d, &md_d, &fail_d);
    }
    // --- static ---
    {
      reset_all();
      CUDA_CHECK(cudaEventRecord(e0, 0));
      for (int it = 0; it < iters; ++it) {
        static_kernel<<<G, T, 0, 0>>>(d_in, d_out, d_val, d_pred_off, d_pred_list,
                                      d_dep, d_succ_off, d_succ_list, d.L, d.W, d.SEG, d.total);
        if (it < iters - 1) reset_all();
      }
      CUDA_CHECK(cudaEventRecord(e1, 0));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_s, e0, e1));
      CUDA_CHECK(cudaMemcpy(h_s, d_out, sizeof(float) * d.W, cudaMemcpyDeviceToHost));
      check(h_s, &md_s, &fail_s);
    }

    double per = (double)iters;
    printf("--- %s ---\n", cases[ci].name);
    printf("  L=%d W=%d FIN=%d tasks=%d  SEG=%d 数据=%.1f MB  边=%d\n",
           d.L, d.W, d.FIN, d.total, d.SEG, ntot * 4.0 / 1e6, pred_off[d.total]);
    printf("  barrier: %9.3f ms   (1.000×)\n", t_b / per);
    printf("  wave  : %9.3f ms   (%6.3f× vs barrier)\n", t_w / per, t_b / t_w);
    printf("  decen : %9.3f ms   (%6.3f× vs barrier)\n", t_d / per, t_b / t_d);
    printf("  static: %9.3f ms   (%6.3f× vs barrier)\n", t_s / per, t_b / t_s);
    printf("  正确性 max rel|Δ|: barrier=%.2e(w=%d) wave=%.2e(w=%d) decen=%.2e(w=%d) static=%.2e(w=%d)\n\n",
           md_b, fail_b, md_w, fail_w, md_d, fail_d, md_s, fail_s);
    fflush(stdout);

    // --- Section C：只在 (L16,W1024,FIN2) 上跑 wave 批量扫描 ---
    if (cases[ci].L == 16 && cases[ci].W == 1024 && cases[ci].FIN == 2) {
      int batches[] = {1, 2, 4, 8, 16, 32};
      printf("--- C: L=16 W=1024 FIN=2 — wave 批量取 task 扫描 ---\n");
      float base = 0;
      for (int bi = 0; bi < 6; ++bi) {
        int b = batches[bi];
        reset_all();
        CUDA_CHECK(cudaEventRecord(e0, 0));
        for (int it = 0; it < iters; ++it) {
          wave_kernel<<<G, T, 0, 0>>>(d_in, d_out, d_val, d_pred_off, d_pred_list,
                                      d_work, d_ldone, d.L, d.W, d.SEG, b);
          if (it < iters - 1) reset_all();
        }
        CUDA_CHECK(cudaEventRecord(e1, 0));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float tw = 0;
        CUDA_CHECK(cudaEventElapsedTime(&tw, e0, e1));
        CUDA_CHECK(cudaMemcpy(h_w, d_out, sizeof(float) * d.W, cudaMemcpyDeviceToHost));
        float md = 0; int f = 0; check(h_w, &md, &f);
        if (bi == 0) base = tw;
        printf("  batch=%3d: %9.3f ms  (%6.3f× vs batch=1)  max rel|Δ|=%.2e fail=%d\n",
               b, tw / per, base / tw, md, f);
        fflush(stdout);
      }
      printf("\n");
    }

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_val); cudaFree(d_pred_off); cudaFree(d_pred_list);
    cudaFree(d_succ_off); cudaFree(d_succ_list); cudaFree(d_dep); cudaFree(d_qbuf);
    cudaFree(d_qready); cudaFree(d_qhead); cudaFree(d_qtail); cudaFree(d_done);
    cudaFree(d_work); cudaFree(d_ldone);
    free(pred_off); free(pred_list); free(succ_off); free(succ_list); free(dep_count);
    free(h_in); free(ref); free(h_qready); free(h_qbuf); free(h_b); free(h_w); free(h_d); free(h_s);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("done.\n");
  return 0;
}
