// 10 ncu 深潜：occupancy 计算、warp stall reasons、roofline、source/SASS 对照。
//
// 用一组「刻意不同」的 kernel 作为样本，每个只在一个维度上有特征：
//   occ<1|2|4|8> : 同一段计算体，用 __launch_bounds__(256, N) 请求不同的驻留 block 数
//                  → 观察寄存器 / occupancy / 溢出的此消彼长
//   latency      : 依赖链式全局访存（pointer chasing）→ 长延迟 stall（long scoreboard）
//   stream       : grid-stride float4 拷贝 → 带宽受限（DRAM 顶线）
//   smem_bar     : shared memory + __syncthreads 循环 → barrier stall
//
// 运行：scripts/run.sh 10-ncu-deep/ncu_deep.cu [iters]
// 剖析：scripts/ncu.sh 10-ncu-deep/ncu_deep.cu --set full --kernel-name regex:occ -- \
//         --only occ
#include "../common/cuda_utils.cuh"

#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// 1) occupancy 样本：同一计算体，请求不同驻留 block 数
//
//    block = 256 线程；64 个独立累加器给足寄存器压力。
//    H100 每 SM：65536 寄存器、最多 2048 线程、最多 32 block。
//    __launch_bounds__(256, MINB) 里第二个参数告诉编译器「至少塞下 MINB 个 block」，
//    编译器会据此压低寄存器用量（代价是 spill）。
// ---------------------------------------------------------------------------
template <int MINB>
__global__ void __launch_bounds__(256, MINB) k_occ(float* __restrict__ out, int iters) {
  float acc[64];
#pragma unroll
  for (int j = 0; j < 64; ++j) acc[j] = threadIdx.x * 0.01f + j;
  const float a = 1.0000001f, b = 0.000001f;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int j = 0; j < 64; ++j) acc[j] = fmaf(acc[j], a, b);
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 64; ++j) s += acc[j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// 2) 长延迟样本：每个线程沿着一条「随机指针链」走 steps 步，
//    每步的地址都依赖上一步的结果 —— 编译器无法预取，只能等访存。
// ---------------------------------------------------------------------------
__global__ void k_latency(const int* __restrict__ idx, float* __restrict__ out, int steps,
                          int n) {
  int p = (blockIdx.x * blockDim.x + threadIdx.x) & (n - 1);
  float s = 0.f;
  for (int i = 0; i < steps; ++i) {
    p = idx[p];       // 依赖上一步的 load，循环无法重叠
    s += (float)p;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// 3) 带宽样本：grid-stride float4 拷贝，稳定压满 HBM
// ---------------------------------------------------------------------------
__global__ void k_stream(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
    out[i] = in[i];
  }
}

// ---------------------------------------------------------------------------
// 4) barrier 样本：每轮写 smem → __syncthreads → 读 → __syncthreads
// ---------------------------------------------------------------------------
__global__ void k_smem_bar(float* __restrict__ out, int iters) {
  __shared__ float s[256];
  const int t = threadIdx.x;
  float v = t * 0.5f;
  for (int i = 0; i < iters; ++i) {
    s[t] = v;
    __syncthreads();
    v = s[(t + 1) & 255] * 0.5f + s[(t + 7) & 255];
    __syncthreads();
  }
  out[blockIdx.x * blockDim.x + t] = v;
}

// ---------------------------------------------------------------------------
// occupancy 计算：手动把 ncu Occupancy 小节的「限制因子」算一遍
// ---------------------------------------------------------------------------
struct OccResult {
  int blocks;                 // 理论上每 SM 能驻留的 block 数
  int threads;                // 对应线程数
  double occupancy;           // threads / maxThreadsPerSM
  int by_regs, by_smem, by_blocks, by_threads;  // 各限制下的 block 上限
  const char* limiter;
};

template <class Kernel>
OccResult calc_occupancy(Kernel k, int block_threads, size_t dyn_smem, const DeviceInfo& d) {
  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, k));
  int max_blocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, k, block_threads,
                                                           dyn_smem));
  OccResult r{};
  const int regs_per_thread = attr.numRegs;
  // 寄存器分配以「warp 粒度、256 个一组」向上取整（H100，见 ncu 文档）
  const int regs_per_warp = ((regs_per_thread * 32 + 255) / 256) * 256;
  const int warps_by_regs = d.regs_per_sm / regs_per_warp;      // 每 SM 可容纳的 warp 数
  r.by_regs = warps_by_regs * 32 / block_threads;              // 换算成 block 数（向下取整）
  const size_t smem_per_block = attr.sharedSizeBytes + dyn_smem;
  r.by_smem = smem_per_block == 0 ? 32 : (int)(d.shared_per_sm / smem_per_block);
  r.by_blocks = 32;  // H100 每 SM 最多 32 个 block
  r.by_threads = d.max_threads_per_sm / block_threads;
  r.blocks = max_blocks;
  r.threads = max_blocks * block_threads;
  r.occupancy = 100.0 * r.threads / d.max_threads_per_sm;

  int best = r.by_regs;
  r.limiter = "registers";
  if (r.by_smem < best) { best = r.by_smem; r.limiter = "shared mem"; }
  if (r.by_blocks < best) { best = r.by_blocks; r.limiter = "block limit"; }
  if (r.by_threads < best) { best = r.by_threads; r.limiter = "threads"; }

  std::printf(
      "  regs=%-3d local=%-4zu smem=%-6zu | blocks/SM: regs=%-2d smem=%-2d blk=%-2d thr=%-2d"
      " -> %d (%.1f%%, limit=%s)\n",
      regs_per_thread, attr.localSizeBytes, smem_per_block, r.by_regs, r.by_smem, r.by_blocks,
      r.by_threads, r.blocks, r.occupancy, r.limiter);
  return r;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? std::atoi(argv[1]) : 20000;
  const char* only = (argc > 2) ? argv[2] : nullptr;  // 只跑指定 kernel（便于 ncu 定向抓取）
  auto want = [&](const char* n) { return only == nullptr || std::strcmp(only, n) == 0; };

  DeviceInfo d = device_info(0);
  print_device_info(d);

  const int blocks = d.sms * 8;  // 1024 个 block
  const int threads = 256;
  const size_t total = (size_t)blocks * threads;

  float* out = nullptr;
  CUDA_CHECK(cudaMalloc(&out, total * sizeof(float)));

  std::printf("\n================ occupancy：手动计算 vs 运行时 API ================\n");
  auto report_occ = [&](const char* name) {
    std::printf("%s\n", name);
  };
  report_occ("k_occ<1>");
  calc_occupancy(k_occ<1>, threads, 0, d);
  report_occ("k_occ<2>");
  calc_occupancy(k_occ<2>, threads, 0, d);
  report_occ("k_occ<4>");
  calc_occupancy(k_occ<4>, threads, 0, d);
  report_occ("k_occ<8>");
  calc_occupancy(k_occ<8>, threads, 0, d);
  report_occ("k_latency");
  calc_occupancy(k_latency, threads, 0, d);
  report_occ("k_stream");
  calc_occupancy(k_stream, threads, 0, d);
  report_occ("k_smem_bar");
  calc_occupancy(k_smem_bar, threads, 0, d);

  std::printf("\n================ 运行时间 ================\n");
  auto run_occ = [&](const char* tag, auto kernel) {
    float ms = (float)bench_ms([&] { kernel<<<blocks, threads>>>(out, iters); }, 3, 20);
    std::printf("  %-14s %8.4f ms\n", tag, ms);
  };
  if (want("occ1")) run_occ("occ<1>", k_occ<1>);
  if (want("occ2")) run_occ("occ<2>", k_occ<2>);
  if (want("occ4")) run_occ("occ<4>", k_occ<4>);
  if (want("occ8")) run_occ("occ<8>", k_occ<8>);

  // latency：随机指针链
  int n = 1 << 20;
  int* idx = nullptr;
  CUDA_CHECK(cudaMalloc(&idx, n * sizeof(int)));
  std::vector<int> h(n);
  for (int i = 0; i < n; ++i) h[i] = (int)(((uint64_t)i * 2654435761u) & (n - 1));
  CUDA_CHECK(cudaMemcpy(idx, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  if (want("latency")) {
    const int lat_blocks = 2048, steps = 256;
    float ms = (float)bench_ms(
        [&] { k_latency<<<lat_blocks, threads>>>(idx, out, steps, n); }, 3, 20);
    std::printf("  %-14s %8.4f ms\n", "latency", ms);
  }

  // stream：128 MB 拷贝
  const size_t bytes = 128ull << 20;
  const size_t n4 = bytes / sizeof(float4);
  float4 *a = nullptr, *b = nullptr;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMemset(a, 1, bytes));
  if (want("stream")) {
    float ms = (float)bench_ms([&] { k_stream<<<blocks, threads>>>(a, b, n4); }, 3, 20);
    double gbps = to_gbps(2.0 * bytes, ms);
    std::printf("  %-14s %8.4f ms  %8.1f GB/s (%.1f%% peak)\n", "stream", ms, gbps,
                100.0 * gbps / d.mem_bw_gbps);
  }

  if (want("smem_bar")) {
    float ms = (float)bench_ms([&] { k_smem_bar<<<blocks, threads>>>(out, iters); }, 3, 20);
    std::printf("  %-14s %8.4f ms\n", "smem_bar", ms);
  }

  CUDA_CHECK(cudaFree(out));
  CUDA_CHECK(cudaFree(idx));
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  return 0;
}
