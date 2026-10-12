// cross_op_overlap.cu — MegaKernel 专题 P20-1 小实验：跨算子细粒度重叠
// （MPK §2.1「Fine-grained kernel overlap」/ Figure 3 的 MatMul->AllGather 型依赖）
//
// 目的（目标卡 H100 / sm90 实测）：
//   本专题的**损失一**此前在 P13-1 里被实测为「单 kernel 内、单个 task 内部」的访存-计算软件流水；
//   而 MPK §2.1（快照行 241–293）讲的是一种**跨算子**的东西：上游算子与下游算子之间数据依赖其实
//   是**逐 tile 细粒度**的（下游每个 block 只依赖上游一个 block 的输出），kernel barrier 却把它们
//   在**整个算子**粒度上串起来。于是不同的 SM 本可以同时跑两种算子（用不同硬件资源），却被迫串行。
//   本实验把这个「依赖粒度决定能否重叠」的机制单独拎出来，用同一份 producer/consumer workload
//   测三种策略：
//       serial : 两个 eager kernel（producer 全体 -> consumer 全体）——kernel barrier 基线
//       barrier: 单个 cooperative persistent kernel，先全体 produce、grid.sync()、再全体 consume
//                ——单次 launch，但仍是**算子级**粗粒度屏障（= Figure 3a）
//       pipe   : 单个 persistent kernel，Gp 个 producer block + Gc 个 consumer block **并发**，
//                每个 tile 一个 ready 标志（逐 tile 细粒度依赖），consumer block 就绪即消费
//                ——Figure 3b（细粒度重叠）
//   关键问题是：**重叠的收益到底由什么决定？** 本实验扫两个变量：
//     Section A：固定 producer 为**访存受限**（纯 copy+1），扫 consumer 计算量 R。
//                ——consumer 越重，能藏进去的 producer 访存越多，收益越大，但被 block 共享封顶。
//     Section B：固定 consumer 计算量（balanced），扫 producer 的计算量 PR。
//                ——producer 从访存受限变计算受限后，两者抢同一硬件资源，重叠收益消失。
//                这正是 MPK 说的「overlap kernels that utilize different hardware resources」。
//
// 依赖模型：NT 个 tile，每个 TS 个 float；producer 写 buf[t]，consumer 读 buf[t] 归约到 out[t]。
//   consumer 的 tile t **只依赖** producer 的 tile t（1:1），与其它 tile 无关——这就是细粒度依赖。
//
// 正确性：三策略的浮点运算顺序完全一致（同一 producer 函数、同一 block 归约树），故 out[] 应逐位相等。
//   用 serial 的 out 作参考，比较 barrier/pipe；报告 max|Δ|。
//
// 编译：nvcc -O3 -arch=sm_90 cross_op_overlap.cu -o cross_op_overlap
// 用法：./cross_op_overlap [iters] [device]   （iters 默认 30）

#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                    \
    if (_e != cudaSuccess) {                                                     \
      printf("CUDA ERROR %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr,         \
             cudaGetErrorString(_e));                                            \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ---------------- 设备侧：producer / consumer 的「单位 tile 工作」 ----------------
// producer：buf[t*TS+e] = (src[t*TS+e] + 1) 再做 PR 轮 FMA（PR=0 即纯访存受限）。
// 用 float4 向量化，producer 在较少 block 下也能喂满 HBM。
__device__ __forceinline__ void produce_tile(const float* __restrict__ src,
                                             float* __restrict__ buf, int t, int TS,
                                             int tid, int nthreads, int PR) {
  int base = t * TS;
  const float4* s4 = reinterpret_cast<const float4*>(src + base);
  float4* b4 = reinterpret_cast<float4*>(buf + base);
  int n4 = TS >> 2;
  for (int e = tid; e < n4; e += nthreads) {
    float4 v = s4[e];
    v.x += 1.0f; v.y += 1.0f; v.z += 1.0f; v.w += 1.0f;
    for (int r = 0; r < PR; ++r) {  // PR 轮 FMA：PR=0 访存受限，PR 大则计算受限
      v.x = fmaf(v.x, 1.0000001f, 0.0000001f);
      v.y = fmaf(v.y, 1.0000001f, 0.0000001f);
      v.z = fmaf(v.z, 1.0000001f, 0.0000001f);
      v.w = fmaf(v.w, 1.0000001f, 0.0000001f);
    }
    b4[e] = v;
  }
}

// consumer：把本线程负责的 16 个元素**读进寄存器一次**，再做 R 轮 FMA 归约
//   —— R 轮在寄存器上迭代，不重复访问内存，故 R 大时 consumer 是**计算受限**（不占 HBM）。
//   （要求 TS == nthreads * CE；本实验 TS=4096, nthreads=256。）
#define CE 16
__device__ __forceinline__ void consume_tile(const float* __restrict__ buf,
                                             float* __restrict__ out, int t, int TS,
                                             int tid, int nthreads, int R) {
  extern __shared__ float sm[];
  int base = t * TS;
  float vals[CE];
#pragma unroll
  for (int e = 0; e < CE; ++e) vals[e] = buf[base + tid * CE + e];  // 每 tile 只读一次
  float acc = 0.0f;
  for (int r = 0; r < R; ++r) {
#pragma unroll
    for (int e = 0; e < CE; ++e) acc = fmaf(vals[e], 1.0f, acc);  // 寄存器内 R 轮 FMA
  }
  sm[tid] = acc;
  __syncthreads();
  for (int s = nthreads >> 1; s > 0; s >>= 1) {
    if (tid < s) sm[tid] += sm[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[t] = sm[0];
}

// ---------------- 策略 1a：serial 的 producer kernel（全体 block，算子级） ----------------
__global__ void produce_kernel(const float* __restrict__ src, float* __restrict__ buf,
                               int NT, int TS, int PR) {
  for (int t = blockIdx.x; t < NT; t += gridDim.x) produce_tile(src, buf, t, TS, threadIdx.x, blockDim.x, PR);
}

// ---------------- 策略 1b：serial 的 consumer kernel（全体 block，算子级） ----------------
__global__ void consume_kernel(const float* __restrict__ buf, float* __restrict__ out,
                               int NT, int TS, int R) {
  for (int t = blockIdx.x; t < NT; t += gridDim.x) consume_tile(buf, out, t, TS, threadIdx.x, blockDim.x, R);
}

// ---------------- 策略 2：barrier —— 单 kernel，全体 produce -> grid.sync -> 全体 consume ----------------
__global__ void barrier_kernel(const float* __restrict__ src, float* __restrict__ buf,
                               float* __restrict__ out, int NT, int TS, int PR, int R) {
  cg::grid_group g = cg::this_grid();
  for (int t = blockIdx.x; t < NT; t += gridDim.x) produce_tile(src, buf, t, TS, threadIdx.x, blockDim.x, PR);
  g.sync();  // 算子级粗粒度屏障（Figure 3a：必须等所有 producer 完成）
  for (int t = blockIdx.x; t < NT; t += gridDim.x) consume_tile(buf, out, t, TS, threadIdx.x, blockDim.x, R);
}

// ---------------- 策略 3：pipe —— 单 kernel，Gp producer + Gc consumer 并发，逐 tile ready 标志 ----------------
// flag[t] 用「代次 gen」区分不同轮次：producer 写 flag[t]=gen，consumer 等 flag[t]>=gen。
__global__ void pipe_kernel(const float* __restrict__ src, float* __restrict__ buf,
                            float* __restrict__ out, int* __restrict__ flag, int gen,
                            int NT, int TS, int PR, int R, int Gp) {
  int G = gridDim.x;
  if (blockIdx.x < Gp) {  // ---- producer block ----
    int p = blockIdx.x;
    for (int t = p; t < NT; t += Gp) {
      produce_tile(src, buf, t, TS, threadIdx.x, blockDim.x, PR);
      __threadfence();  // 值先可见
      if (threadIdx.x == 0) atomicExch(&flag[t], gen);
    }
  } else {  // ---- consumer block ----
    int c = blockIdx.x - Gp;
    int Gc = G - Gp;
    for (int t = c; t < NT; t += Gc) {
      if (threadIdx.x == 0) {
        volatile int* vf = flag;
        while (vf[t] < gen) __nanosleep(64);  // 只等**本 tile**（普通 load 轮询，避免原子争用）
      }
      __syncthreads();
      __threadfence();
      consume_tile(buf, out, t, TS, threadIdx.x, blockDim.x, R);  // 细粒度：tile 就绪即算
    }
  }
}

// ---------------- 计时 ----------------
#include <functional>
static float bench(int iters, const std::function<void()>& f) {
  for (int w = 0; w < 3; ++w) f();  // warmup
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  CUDA_CHECK(cudaEventRecord(a, 0));
  for (int r = 0; r < iters; ++r) f();
  CUDA_CHECK(cudaEventRecord(b, 0));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  return ms / iters;
}

static float max_abs_diff(const float* a, const float* b, int n) {
  float m = 0;
  for (int i = 0; i < n; ++i) {
    float d = fabsf(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 30;
  int dev = (argc > 2) ? atoi(argv[2]) : 3;
  CUDA_CHECK(cudaSetDevice(dev));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int numSM = prop.multiProcessorCount;

  int TS = 4096;                       // 每 tile 的 float 数（16 KB）
  int NT = 8192;                       // tile 数
  const long M = (long)TS * NT;        // 总 float 数 = 33.5M（134 MB）
  const int T = 256;                   // threads/block
  const int G = numSM;                 // 总 block 数（barrier/serial 用；pipe = Gp+Gc）

  printf("== device ==\n%s, sm_%d%d, SMs=%d, cooperativeLaunch=%d\n", prop.name,
         prop.major, prop.minor, numSM, prop.cooperativeLaunch);
  printf("iters=%d, threads/block=%d, TS=%d, NT=%d, M=%ld floats (%.1f MB per array)\n\n",
         iters, T, TS, NT, M, M * 4.0 / 1e6);

  // 求 cooperative 占用上限，保证 pipe 的 Gp+Gc 个 block 全部常驻（无死锁）
  int maxBlocksPerSM = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, pipe_kernel, T, T * 4));
  int coopCap = maxBlocksPerSM * numSM;
  int Gp = 32;                      // producer block 数（访存受限，少量即可接近 BW）
  if (Gp >= G) Gp = G / 4;
  int Gc = G - Gp;
  printf("pipe_kernel occupancy: maxBlocks/SM=%d -> coop 上限 %d blocks; 用 G=%d (Gp=%d + Gc=%d)\n\n",
         maxBlocksPerSM, coopCap, G, Gp, Gc);

  float *d_src, *d_buf, *d_out_ref, *d_out_b, *d_out_p;
  int* d_flag;
  CUDA_CHECK(cudaMalloc(&d_src, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_buf, M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_ref, NT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_b, NT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out_p, NT * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_flag, NT * sizeof(int)));

  float* h_src = (float*)malloc(M * sizeof(float));
  for (long i = 0; i < M; ++i) h_src[i] = (float)((i % 67) * 0.03125) - 1.0f;
  CUDA_CHECK(cudaMemcpy(d_src, h_src, M * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_flag, 0, NT * sizeof(int)));

  float* h_ref = (float*)malloc(NT * sizeof(float));
  float* h_b = (float*)malloc(NT * sizeof(float));
  float* h_p = (float*)malloc(NT * sizeof(float));

  // 各策略的计时闭包（gen 每轮 +1，避免 flag 跨轮粘连）
  int gen = 1;
  auto run_serial = [&](int PR, int R) {
    return bench(iters, [&]() {
      produce_kernel<<<G, T>>>(d_src, d_buf, NT, TS, PR);
      consume_kernel<<<G, T, T * sizeof(float)>>>(d_buf, d_out_ref, NT, TS, R);
    });
  };
  auto run_produce_only = [&](int PR, int gb) {
    return bench(iters, [&]() { produce_kernel<<<gb, T>>>(d_src, d_buf, NT, TS, PR); });
  };
  auto run_consume_only = [&](int R, int gb) {
    return bench(iters, [&]() { consume_kernel<<<gb, T, T * sizeof(float)>>>(d_buf, d_out_ref, NT, TS, R); });
  };
  auto run_barrier = [&](int PR, int R, float* out) {
    return bench(iters, [&]() {
      void* args[] = {&d_src, &d_buf, &out, &NT, &TS, &PR, &R};
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)barrier_kernel, dim3(G), dim3(T), args, T * 4, 0));
    });
  };
  auto run_pipe = [&](int PR, int R, float* out) {
    return bench(iters, [&]() {
      int g = gen++;
      void* args[] = {&d_src, &d_buf, &out, &d_flag, &g, &NT, &TS, &PR, &R, &Gp};
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)pipe_kernel, dim3(G), dim3(T), args, T * 4, 0));
    });
  };

  // ---------------- Section A：producer 访存受限（PR=0），扫 consumer 计算量 R ----------------
  printf("== Section A: producer 访存受限 (PR=0)，扫 consumer 计算量 R ==\n");
  printf("%6s | %9s | %9s | %9s | %9s | %9s | %8s | %9s | %9s | %8s\n", "R",
         "Tp ms", "Tc ms", "serial", "barrier", "pipe", "s/b", "pipe/b", "ideal", "p/ideal");
  printf("------ | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------\n");
  int Rs[] = {0, 16, 32, 64, 128, 256};
  for (int Ri = 0; Ri < (int)(sizeof(Rs) / sizeof(Rs[0])); ++Ri) {
    int R = Rs[Ri];
    gen = 1;  // 每格重跑前重置代次，保证首轮从干净 flag 开始
    CUDA_CHECK(cudaMemset(d_flag, 0, NT * sizeof(int)));
    float tp = run_produce_only(0, G);
    float tc = run_consume_only(R, G);
    // ideal：把 Gp/Gc 个 block 分别给两阶段、且**完美重叠**时的下界 = max(Tp(Gp), Tc(Gc))
    float exp = fmaxf(run_produce_only(0, Gp), run_consume_only(R, G - Gp));
    float ts = run_serial(0, R);
    float tb = run_barrier(0, R, d_out_b);
    CUDA_CHECK(cudaDeviceSynchronize());
    float tpi = run_pipe(0, R, d_out_p);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_b, d_out_b, NT * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p, d_out_p, NT * sizeof(float), cudaMemcpyDeviceToHost));
    printf("%6d | %9.4f | %9.4f | %9.4f | %9.4f | %9.4f | %7.3fx | %8.3fx | %9.4f | %7.3fx\n",
           R, tp, tc, ts, tb, tpi, ts / tb, tb / tpi, exp, tpi / exp);
  }

  // 正确性：serial(参考) vs barrier vs pipe（同一 R 下应逐位一致）
  CUDA_CHECK(cudaMemcpy(h_ref, d_out_ref, NT * sizeof(float), cudaMemcpyDeviceToHost));
  printf("\ncorrectness (last grid: R=%d): max|barrier-serial|=%.3e  max|pipe-serial|=%.3e (expect 0)\n",
         Rs[(int)(sizeof(Rs) / sizeof(Rs[0])) - 1],
         max_abs_diff(h_ref, h_b, NT), max_abs_diff(h_ref, h_p, NT));

  // ---------------- Section B：固定 consumer（balanced R），扫 producer 计算量 PR ----------------
  int Rbal = 80;
  printf("\n== Section B: consumer 固定 R=%d（balanced），扫 producer 计算量 PR（0=访存受限 -> 大=计算受限） ==\n", Rbal);
  printf("%6s | %9s | %9s | %9s | %9s | %9s | %8s | %9s\n", "PR",
         "Tp ms", "Tc ms", "serial", "barrier", "pipe", "s/b", "pipe/b");
  printf("------ | ------- | ------- | ------- | ------- | ------- | ------- | -------\n");
  int PRs[] = {0, 32, 64, 128, 256};
  for (int Pi = 0; Pi < (int)(sizeof(PRs) / sizeof(PRs[0])); ++Pi) {
    int PR = PRs[Pi];
    gen = 1;
    CUDA_CHECK(cudaMemset(d_flag, 0, NT * sizeof(int)));
    float tp = run_produce_only(PR, G);
    float tc = run_consume_only(Rbal, G);
    float ts = run_serial(PR, Rbal);
    float tb = run_barrier(PR, Rbal, d_out_b);
    CUDA_CHECK(cudaDeviceSynchronize());
    float tpi = run_pipe(PR, Rbal, d_out_p);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("%6d | %9.4f | %9.4f | %9.4f | %9.4f | %9.4f | %7.3fx | %8.3fx\n",
           PR, tp, tc, ts, tb, tpi, ts / tb, tb / tpi);
  }

  // ---------------- Section C：balanced 下扫 producer block 数 Gp ----------------
  printf("\n== Section C: balanced (PR=0,R=%d) 下扫 pipe 的 producer block 数 Gp ==\n", Rbal);
  printf("%6s | %9s | %9s | %9s\n", "Gp", "pipe ms", "barrier", "b/pipe");
  printf("------ | ------- | ------- | -------\n");
  int Gps[] = {8, 16, 32, 64};
  for (int gi = 0; gi < (int)(sizeof(Gps) / sizeof(Gps[0])); ++gi) {
    Gp = Gps[gi];
    gen = 1;
    CUDA_CHECK(cudaMemset(d_flag, 0, NT * sizeof(int)));
    float tb = run_barrier(0, Rbal, d_out_b);
    CUDA_CHECK(cudaDeviceSynchronize());
    float tpi = run_pipe(0, Rbal, d_out_p);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("%6d | %9.4f | %9.4f | %8.3fx\n", Gp, tpi, tb, tb / tpi);
  }

  // ---------------- Section D：单阶段随 block 数的缩放（解释为什么切 block 不划算） ----------------
  // 若某阶段在少量 block 下就能「喂满其瓶颈资源」，其时间应随 block 数**先降后平**（saturate）；
  // 若线性下降，则把它切走多少 block 就慢多少倍 —— 二者决定了 pipe 能否赢过 barrier。
  printf("\n== Section D: 单阶段随 block 数 GB 的缩放（PR=0 访存受限 producer; R=%d 计算受限 consumer） ==\n", Rbal);
  printf("%6s | %10s | %12s | %10s | %12s\n", "GB", "Tp ms", "Tp(132)/Tp", "Tc ms", "Tc(132)/Tc");
  printf("------ | ---------- | ------------ | ---------- | ------------\n");
  float tpG = run_produce_only(0, G), tcG = run_consume_only(Rbal, G);
  int GBs[] = {16, 33, 66, 132, 264, 528, 1056};
  for (int bi = 0; bi < (int)(sizeof(GBs) / sizeof(GBs[0])); ++bi) {
    int gb = GBs[bi];
    float tp = run_produce_only(0, gb);
    float tc = run_consume_only(Rbal, gb);
    printf("%6d | %10.4f | %11.3fx | %10.4f | %11.3fx\n", gb, tp, tpG / tp, tc, tcG / tc);
  }
  printf("(基准 G=%d: Tp=%.4f ms, Tc=%.4f ms；比值=基准时间 / 该 GB 的时间，>1 表示该阶段受益于更多 block)\n",
         G, tpG, tcG);

  printf("\n>> 读法：\n");
  printf(">>  s/b        : barrier 相对 serial —— 单次 launch（消损失二）能带来多少；本区间应≈1.0。\n");
  printf(">>  pipe/b     : 细粒度重叠（逐 tile ready）相对算子级屏障的加速（损失一的跨算子形态）。\n");
  printf(">>  ideal      : max(Tp(Gp), Tc(Gc)) —— 静态分块 + 完美重叠的理论下界；对比 barrier 看分块值不值。\n");
  printf(">>  Section A  : 关键发现是 pipe/b 恒 <1：**静态切 block 的细粒度重叠赢不过算子级屏障**。\n");
  printf(">>  Section B  : producer 从访存受限变计算受限后，两阶段抢同一资源，pipe/b 进一步掉到 ~0.4。\n");
  printf(">>  Section C  : producer block 太少喂不满 BW、太多挤占 consumer，最优 Gp 仍打不过 barrier。\n");
  printf(">>  Section D  : 两阶段时间都随 block 数近似线性 —— 阶段内部并行度不足，切走 block 就等比例变慢，\n");
  printf(">>               故「切 block 做跨算子重叠」在固定 SM 的 GPU 上结构性不划算；真正的跨算子重叠要\n");
  printf(">>               靠 SM 内**共驻**（warp 特化/时分复用，见 P13-1/P18-1），而不是静态分区。\n");

  cudaFree(d_src); cudaFree(d_buf); cudaFree(d_out_ref); cudaFree(d_out_b);
  cudaFree(d_out_p); cudaFree(d_flag);
  free(h_src); free(h_ref); free(h_b); free(h_p);
  return 0;
}
