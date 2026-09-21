// 18 DSA 稀疏注意力（一）：lightning indexer + exact top-k
//
// DeepSeek-V4 / V3.2-Exp 的 DSA（DeepSeek Sparse Attention）把稠密 attention
// 拆成两步：
//   1) lightning indexer：用少量 index head（V4-Pro: HI=64, D=128）给每个 query
//      对每个历史 key 打一个「索引分」
//          I[t,s] = sum_j w[t,j] * ReLU( q_index[t,j,:] . k_index[s,:] )
//      （prod 用 FP8，这里用 bf16 做等价 benchmark），再取 top-k（V4-Pro: 1024）；
//   2) 只对被选中的 k 个 key 做真正的 MLA attention。
//
// 本篇聚焦最独特、也最容易成为新瓶颈的两个算子：indexer 打分 与 exact top-k。
// 关键结论：
//   * indexer 是纯算力受限的 GEMM（HI 个小 GEMM 共享 K），但 head 维要循环，
//     Q 会被反复重读 —— 我们把 Q tile 常驻 L2 后可以做到 XX TFLOPS；
//   * exact top-k 不需要「全排序」：用 radix-select（逐 8-bit 位）4 趟即可定位
//     第 k 大的「有序整数」阈值，再把 > 阈值 与 == 阈值 的补齐；
//   * top-k 是纯带宽受限；把整行放进 smem 后，4 趟扫描全在片内，省掉 5x 的
//     global 往返。
//
// 运行：scripts/run.sh 18-dsa-sparse/dsa.cu [S] [mode]
//   mode = all | idx | topk | scal
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

using bf16 = __nv_bfloat16;

// DeepSeek-V4-Pro DSA indexer 口径（/ssd/models/DeepSeek-V4-Pro/config.json）
constexpr int ID = 128;   // index_head_dim
constexpr int IHI = 64;   // index_n_heads
constexpr int TOPK = 1024;  // index_topk

// ---------------------------------------------------------------------------
// PTX helpers（与 16 篇同一套）
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack2(float x, float y) {
  __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}

// ===========================================================================
// 1) lightning indexer 打分：scores[t][s] = sum_j w[t,j] * ReLU(q[t,j] . k[s])
// ===========================================================================
// 每个 block 负责 [BM 个 query] x [BN 个 key] 的 tile。K tile 常驻 smem，
// 循环 HI 个 head：每个 head 把 Q_j tile 搬进 smem，做 D=128 的 mma（8 个 k16），
// epilogue 里 ReLU + 乘 w，累加到 fp32 的 acc（跨 head 求和）。
//
// 关键优化 HG：B 操作数（K）对所有 head 相同，默认每个 head 都把整个 K tile 用
// ldmatrix 重读一遍（L1/TEX 85% 的元凶）。把 HG 个 head 一起算：每个 (kx,nb) 只
// 加载一次 B，喂给 HG 个 A，K 的 ldmatrix 流量降到 1/HG。
template <int BM, int BN, int WARPS, int HG>
__global__ void __launch_bounds__(WARPS * 32) indexer_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ kk, const float* __restrict__ w,
    float* __restrict__ scores, int S) {
  constexpr int DKP = ID + 8;  // +8 bf16 padding 消 ldmatrix bank conflict
  constexpr int NT = BN / 8;   // n8 tile 数
  extern __shared__ bf16 sm[];
  bf16(*ks)[DKP] = reinterpret_cast<bf16(*)[DKP]>(sm);                    // BN x DKP
  bf16(*qs)[DKP] = reinterpret_cast<bf16(*)[DKP]>(sm + BN * DKP);         // HG x BM x DKP

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int T = WARPS * 32;
  const int q0 = blockIdx.y * BM, k0 = blockIdx.x * BN;  // x=kv 最快变化，Q tile 留 L2

  // 载入 K tile
  for (int i = tid; i < BN * ID / 8; i += T) {
    const int row = i / (ID / 8), c8 = (i % (ID / 8)) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    const int gk = k0 + row;
    if (gk < S) v = *reinterpret_cast<const uint4*>(&kk[(size_t)gk * ID + c8]);
    *reinterpret_cast<uint4*>(&ks[row][c8]) = v;
  }
  __syncthreads();

  float acc[NT][4];
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int a = 0; a < 4; ++a) acc[t][a] = 0.f;

  const int row0 = warp * 16 + (lane >> 2);  // 本 warp 的 m16 行（c0/c1）
  const int n_off = (lane & 7) + ((lane >> 4) & 1) * 8;
  const int k_off = ((lane >> 3) & 1) * 8;

#pragma unroll 1
  for (int j0 = 0; j0 < IHI; j0 += HG) {
    // HG 个 Q tile: q[(gq)*IHI*ID + (j0+h)*ID + d]
    for (int i = tid; i < HG * BM * ID / 8; i += T) {
      const int h = i / (BM * ID / 8);
      const int r = i % (BM * ID / 8);
      const int row = r / (ID / 8), c8 = (r % (ID / 8)) * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      const int gq = q0 + row;
      if (gq < S)
        v = *reinterpret_cast<const uint4*>(&q[((size_t)gq * IHI + (j0 + h)) * ID + c8]);
      *reinterpret_cast<uint4*>(&qs[h * BM + row][c8]) = v;
    }
    __syncthreads();

    float w0[HG], w1[HG];
#pragma unroll
    for (int h = 0; h < HG; ++h) {
      w0[h] = (q0 + row0 < S) ? w[(size_t)(q0 + row0) * IHI + j0 + h] : 0.f;
      w1[h] = (q0 + row0 + 8 < S) ? w[(size_t)(q0 + row0 + 8) * IHI + j0 + h] : 0.f;
    }

    float sacc[HG][NT][4];
#pragma unroll
    for (int h = 0; h < HG; ++h)
#pragma unroll
      for (int t = 0; t < NT; ++t)
#pragma unroll
        for (int a = 0; a < 4; ++a) sacc[h][t][a] = 0.f;

#pragma unroll
    for (int kx = 0; kx < ID / 16; ++kx) {
      uint32_t A[HG][4];
#pragma unroll
      for (int h = 0; h < HG; ++h)
        ldmatrix_x4(smem_u32(&qs[h * BM + warp * 16 + (lane & 15)][kx * 16 + (lane >> 4) * 8]),
                    A[h]);
#pragma unroll
      for (int nb = 0; nb < BN / 16; ++nb) {
        uint32_t Bf[4];
        ldmatrix_x4(smem_u32(&ks[nb * 16 + n_off][kx * 16 + k_off]), Bf);
#pragma unroll
        for (int h = 0; h < HG; ++h) {
          mma16816(sacc[h][nb * 2], A[h], Bf);
          mma16816(sacc[h][nb * 2 + 1], A[h], Bf + 2);
        }
      }
    }
    // epilogue: ReLU + 权重 + 累加
#pragma unroll
    for (int h = 0; h < HG; ++h)
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        acc[t][0] += w0[h] * fmaxf(sacc[h][t][0], 0.f);
        acc[t][1] += w0[h] * fmaxf(sacc[h][t][1], 0.f);
        acc[t][2] += w1[h] * fmaxf(sacc[h][t][2], 0.f);
        acc[t][3] += w1[h] * fmaxf(sacc[h][t][3], 0.f);
      }
    __syncthreads();
  }

  const int t0 = q0 + row0, t1 = t0 + 8;
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int col = k0 + t * 8 + (lane & 3) * 2;
    if (t0 < S) {
      scores[(size_t)t0 * S + col] = acc[t][0];
      scores[(size_t)t0 * S + col + 1] = acc[t][1];
    }
    if (t1 < S) {
      scores[(size_t)t1 * S + col] = acc[t][2];
      scores[(size_t)t1 * S + col + 1] = acc[t][3];
    }
  }
}

// 标量基线（仅用于小 S 对拍/参考性能）
__global__ void indexer_scalar_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kk,
                                      const float* __restrict__ w, float* __restrict__ scores,
                                      int S) {
  const int t = blockIdx.y * blockDim.y + threadIdx.y;
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= S || s >= S) return;
  float acc = 0.f;
  for (int j = 0; j < IHI; ++j) {
    float dot = 0.f;
    for (int d = 0; d < ID; ++d)
      dot += __bfloat162float(q[((size_t)t * IHI + j) * ID + d]) *
             __bfloat162float(kk[(size_t)s * ID + d]);
    acc += w[(size_t)t * IHI + j] * fmaxf(dot, 0.f);
  }
  scores[(size_t)t * S + s] = acc;
}

// ===========================================================================
// 2) exact top-k：radix select
// ===========================================================================
// 把 float 映射成单调的 uint32（order preserving），从最高 8 bit 开始逐趟定位
// 「第 k 大的值」所在的 digit；4 趟后 prefix 就是第 k 大的「有序值」。
// 再收集所有 > prefix 的下标，并用 == prefix 的补齐到 k 个。
__device__ __forceinline__ uint32_t order_float(float f) {
  uint32_t u = __float_as_uint(f);
  return u ^ (((int32_t)u >> 31) | 0x80000000u);  // 自逆
}
__device__ __forceinline__ float unorder_float(uint32_t o) {
  return (o & 0x80000000u) ? __uint_as_float(o & 0x7fffffffu) : __uint_as_float(~o);
}

constexpr int RBITS = 8;
constexpr int RBUCK = 1 << RBITS;

// 整行常驻 smem 的版本：4 趟扫描全在片内，省掉 5x global 往返。
// 每 warp 私有直方图（NH 份）消 atomicAdd 竞争；NH+1 份的第 0 份存规约结果。
template <int T, int NH>
__global__ void topk_radix_smem_kernel(const float* __restrict__ scores, int S, int k,
                                       int* __restrict__ out_idx, float* __restrict__ out_val) {
  extern __shared__ unsigned int urow[];            // S x u32
  unsigned int* shist = urow + S;                   // (NH+1) x RBUCK
  unsigned int* total = shist;                      // 规约结果
  const int row = blockIdx.x;
  const float* r = scores + (size_t)row * S;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  unsigned int* myhist = shist + (1 + (wid % NH)) * RBUCK;

  for (int i = tid; i < S; i += T) urow[i] = order_float(r[i]);
  __syncthreads();

  unsigned int prefix = 0;
  int remaining = k;
#pragma unroll
  for (int pass = 0; pass < 32 / RBITS; ++pass) {
    const int shift = 32 - RBITS * (pass + 1);
    for (int i = tid; i < (NH + 1) * RBUCK; i += T) shist[i] = 0;
    __syncthreads();
    for (int i = tid; i < S; i += T) {
      const unsigned int u = urow[i];
      const int hi = shift + RBITS;
      if (hi >= 32 || ((u >> hi) == (prefix >> hi)))
        atomicAdd(&myhist[(u >> shift) & (RBUCK - 1)], 1u);
    }
    __syncthreads();
    for (int b = tid; b < RBUCK; b += T) {
      unsigned int acc = 0;
#pragma unroll
      for (int h = 0; h < NH; ++h) acc += shist[(1 + h) * RBUCK + b];
      total[b] = acc;
    }
    __syncthreads();
    int acc = 0, d = RBUCK - 1;
#pragma unroll
    for (int b = RBUCK - 1; b >= 0; --b) {
      if (acc + (int)total[b] >= remaining) {
        d = b;
        break;
      }
      acc += total[b];
    }
    remaining -= acc;
    prefix |= (unsigned int)d << shift;
    __syncthreads();
  }

  __shared__ int counter;
  if (tid == 0) counter = 0;
  __syncthreads();
  for (int i = tid; i < S; i += T) {
    const unsigned int u = urow[i];
    if (u > prefix) {
      const int slot = atomicAdd(&counter, 1);
      out_idx[(size_t)row * k + slot] = i;
      if (out_val) out_val[(size_t)row * k + slot] = unorder_float(u);
    }
  }
  __syncthreads();
  for (int i = tid; i < S; i += T) {
    const unsigned int u = urow[i];
    if (u == prefix) {
      const int slot = atomicAdd(&counter, 1);
      if (slot < k) {
        out_idx[(size_t)row * k + slot] = i;
        if (out_val) out_val[(size_t)row * k + slot] = unorder_float(u);
      }
    }
  }
}

// 行放不下 smem 时的流式版本：每趟都从 global 读一遍（5 趟）。同样 per-warp 直方图。
template <int T, int NH>
__global__ void topk_radix_global_kernel(const float* __restrict__ scores, int S, int k,
                                         int* __restrict__ out_idx) {
  __shared__ unsigned int shist[(NH + 1) * RBUCK];
  unsigned int* total = shist;
  const int row = blockIdx.x;
  const float* r = scores + (size_t)row * S;
  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  unsigned int* myhist = shist + (1 + (wid % NH)) * RBUCK;

  unsigned int prefix = 0;
  int remaining = k;
#pragma unroll
  for (int pass = 0; pass < 32 / RBITS; ++pass) {
    const int shift = 32 - RBITS * (pass + 1);
    for (int i = tid; i < (NH + 1) * RBUCK; i += T) shist[i] = 0;
    __syncthreads();
    for (int i = tid; i < S; i += T) {
      const unsigned int u = order_float(r[i]);
      const int hi = shift + RBITS;
      if (hi >= 32 || ((u >> hi) == (prefix >> hi)))
        atomicAdd(&myhist[(u >> shift) & (RBUCK - 1)], 1u);
    }
    __syncthreads();
    for (int b = tid; b < RBUCK; b += T) {
      unsigned int acc = 0;
#pragma unroll
      for (int h = 0; h < NH; ++h) acc += shist[(1 + h) * RBUCK + b];
      total[b] = acc;
    }
    __syncthreads();
    int acc = 0, d = RBUCK - 1;
#pragma unroll
    for (int b = RBUCK - 1; b >= 0; --b) {
      if (acc + (int)total[b] >= remaining) {
        d = b;
        break;
      }
      acc += total[b];
    }
    remaining -= acc;
    prefix |= (unsigned int)d << shift;
    __syncthreads();
  }

  __shared__ int counter;
  if (tid == 0) counter = 0;
  __syncthreads();
  for (int i = tid; i < S; i += T) {
    const unsigned int u = order_float(r[i]);
    if (u > prefix) {
      const int slot = atomicAdd(&counter, 1);
      out_idx[(size_t)row * k + slot] = i;
    }
  }
  __syncthreads();
  for (int i = tid; i < S; i += T) {
    const unsigned int u = order_float(r[i]);
    if (u == prefix) {
      const int slot = atomicAdd(&counter, 1);
      if (slot < k) out_idx[(size_t)row * k + slot] = i;
    }
  }
}

// ---------------------------------------------------------------------------
// host references
// ---------------------------------------------------------------------------
static void indexer_ref(int t, int S, const std::vector<bf16>& q, const std::vector<bf16>& kk,
                        const std::vector<float>& w, std::vector<float>& row) {
  row.assign(S, 0.f);
  for (int s = 0; s < S; ++s) {
    float acc = 0.f;
    for (int j = 0; j < IHI; ++j) {
      float dot = 0.f;
      for (int d = 0; d < ID; ++d)
        dot += __bfloat162float(q[((size_t)t * IHI + j) * ID + d]) *
               __bfloat162float(kk[(size_t)s * ID + d]);
      acc += w[(size_t)t * IHI + j] * std::max(dot, 0.f);
    }
    row[s] = acc;
  }
}

// ---------------------------------------------------------------------------
// device data fill
// ---------------------------------------------------------------------------
__global__ void fill_bf16_kernel(bf16* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned h = (unsigned)(i * 2654435761u + seed);
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  p[i] = __float2bfloat16((float)(h & 0xffff) / 65536.f - 0.5f);
}
__global__ void fill_f32_kernel(float* p, size_t n, unsigned seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned h = (unsigned)(i * 2246822519u + seed);
  h ^= h >> 13;
  h *= 3266489917u;
  h ^= h >> 16;
  p[i] = (float)(h & 0xffff) / 65536.f;
}

template <class T>
static void fill_dev(T* p, size_t n, unsigned seed) {
  if constexpr (std::is_same_v<T, float>)
    fill_f32_kernel<<<div_up((int)((n + 255) / 256), 1), 256>>>((float*)p, n, seed);
  else
    fill_bf16_kernel<<<div_up((int)((n + 255) / 256), 1), 256>>>((bf16*)p, n, seed);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int S = (argc > 1) ? std::atoi(argv[1]) : 4096;
  const char* mode = (argc > 2) ? argv[2] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);

  const double idx_flops = 2.0 * (double)S * S * IHI * ID;  // score 计算 FLOPs
  std::printf("\nDSA lightning indexer + top-k: S=%d  HI=%d D=%d topk=%d\n", S, IHI, ID, TOPK);
  std::printf("indexer FLOPs = %.2f GFLOP ; scores = %.2f GB\n\n", idx_flops / 1e9,
              (double)S * S * 4 / 1e9);

  bf16 *q, *kk;
  float *w, *scores;
  int* out_idx;
  float* out_val;
  CUDA_CHECK(cudaMalloc(&q, (size_t)S * IHI * ID * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&kk, (size_t)S * ID * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&w, (size_t)S * IHI * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&scores, (size_t)S * S * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&out_idx, (size_t)S * TOPK * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out_val, (size_t)S * TOPK * sizeof(float)));

  fill_dev<bf16>(q, (size_t)S * IHI * ID, 1);
  fill_dev<bf16>(kk, (size_t)S * ID, 2);
  fill_dev<float>(w, (size_t)S * IHI, 3);

  auto want = [&](const char* name) {
    return std::strcmp(mode, "all") == 0 || std::strcmp(mode, name) == 0;
  };

  constexpr int BM = 64, BN = 128, WARPS = 4;
  const size_t idx_shm = (size_t)(2 * BM + BN) * (ID + 8) * sizeof(bf16);
  auto idx_fn = indexer_kernel<BM, BN, WARPS, 2>;
  CUDA_CHECK(cudaFuncSetAttribute(idx_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  (int)idx_shm));
  dim3 idx_grid(div_up(S, BN), div_up(S, BM));  // x = kv 块，y = query 块
  auto run_idx = [&] {
    idx_fn<<<idx_grid, WARPS * 32, idx_shm>>>(q, kk, w, scores, S);
  };

  // ---- indexer 对拍（只查前几个 query 的若干列，够了） ----
  if ((want("idx") || want("all")) && (double)S * S * 4 > 5e9) {
    run_idx();
    CUDA_CHECK_LAST();
    std::printf("[indexer TC ] correctness check skipped (scores %.1f GB too big for host)\n",
                (double)S * S * 4 / 1e9);
  } else if (want("idx") || want("all")) {
    run_idx();
    CUDA_CHECK_LAST();
    std::vector<bf16> hq((size_t)S * IHI * ID), hk((size_t)S * ID);
    std::vector<float> hw((size_t)S * IHI), hs((size_t)S * S);
    CUDA_CHECK(cudaMemcpy(hq.data(), q, hq.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hk.data(), kk, hk.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hw.data(), w, hw.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hs.data(), scores, hs.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double maxerr = 0, maxref = 0;
    std::vector<float> ref;
    const int qt[4] = {0, S / 3, S / 2, S - 1};
    for (int qi = 0; qi < 4; ++qi) {
      const int t = qt[qi];
      if (t < 0 || t >= S) continue;
      indexer_ref(t, S, hq, hk, hw, ref);
      for (int s = 0; s < S; s += std::max(1, S / 17)) {
        const double e = std::fabs((double)hs[(size_t)t * S + s] - ref[s]);
        maxerr = std::max(maxerr, e);
        maxref = std::max(maxref, (double)std::fabs(ref[s]));
      }
    }
    std::printf("[indexer TC ] max_abs_err=%.3e (ref~%.3f) %s\n", maxerr, maxref,
                maxerr / std::max(maxref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  }

  if (want("idx") || want("all")) {
#define LAUNCH_IDX(HG, BNT, WRP)                                                          \
  {                                                                                       \
    constexpr int BMx = (WRP) * 16;                                                       \
    const size_t shm = (size_t)((HG) * BMx + (BNT)) * (ID + 8) * sizeof(bf16);            \
    CUDA_CHECK(cudaFuncSetAttribute(indexer_kernel<BMx, (BNT), (WRP), (HG)>,              \
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));\
    dim3 g(div_up(S, BNT), div_up(S, BMx));                                               \
    auto run = [&] {                                                                      \
      indexer_kernel<BMx, (BNT), (WRP), (HG)><<<g, (WRP) * 32, shm>>>(q, kk, w, scores, S); \
    };                                                                                    \
    run();                                                                                \
    CUDA_CHECK_LAST();                                                                    \
    double t = bench_ms(run, 3, 10);                                                      \
    std::printf("[indexer BM=%d HG=%d BN=%d W=%d] %8.4f ms  %8.2f TFLOPS (%5.2f%% peak)\n", \
                BMx, HG, BNT, WRP, t, to_tflops(idx_flops, t),                            \
                100.0 * to_tflops(idx_flops, t) / 989.0);                                 \
  }
    LAUNCH_IDX(1, 64, 4)
    LAUNCH_IDX(2, 64, 4)
    LAUNCH_IDX(1, 128, 4)
    LAUNCH_IDX(2, 128, 4)
    LAUNCH_IDX(4, 128, 4)
#undef LAUNCH_IDX
  }

  if (want("scal") && S > 8192) {
    std::printf("[indexer scal] skipped (S>8192, 标量基线太慢)\n");
  } else if (want("scal")) {
    dim3 blk(16, 16);
    dim3 grd(div_up(S, 16), div_up(S, 16));
    auto run_scal = [&] { indexer_scalar_kernel<<<grd, blk>>>(q, kk, w, scores, S); };
    run_scal();
    CUDA_CHECK_LAST();
    double t = bench_ms(run_scal, 2, 5);
    std::printf("[indexer scal] %8.4f ms  %8.2f TFLOPS\n", t, to_tflops(idx_flops, t));
  }

  // ---- top-k ----
  // 用随机分布填充 scores（top-k 只关心选择，与 indexer 数值无关），
  // 但先做一次正确性检查：拿第 0 行的 top-k 值集合与 CPU 精确解对比。
  if (want("topk") || want("all")) {
    const int CS = std::min(S, 8192);  // 检查用的小行数/行宽
    std::vector<float> hs((size_t)CS * CS);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hs) v = dist(rng);
    CUDA_CHECK(cudaMemcpy(scores, hs.data(), hs.size() * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int TT = 1024, NH = TT / 32;
    const size_t cshm = (size_t)CS * 4 + (NH + 1) * RBUCK * 4;
    CUDA_CHECK(cudaFuncSetAttribute(topk_radix_smem_kernel<TT, NH>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cshm));
    topk_radix_smem_kernel<TT, NH><<<CS, TT, cshm>>>(scores, CS, std::min(TOPK, CS), out_idx,
                                                     out_val);
    CUDA_CHECK_LAST();
    std::vector<int> hidx((size_t)CS * std::min(TOPK, CS));
    std::vector<float> hval((size_t)CS * std::min(TOPK, CS));
    const int kc = std::min(TOPK, CS);
    CUDA_CHECK(cudaMemcpy(hidx.data(), out_idx, hidx.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hval.data(), out_val, hval.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double maxerr = 0;
    for (int row = 0; row < 3; ++row) {
      std::vector<float> r(hs.begin() + (size_t)row * CS, hs.begin() + (size_t)(row + 1) * CS);
      std::vector<float> ref = r;
      std::nth_element(ref.begin(), ref.begin() + (kc - 1), ref.end(), std::greater<float>());
      const float thresh = ref[kc - 1];
      // 检查：选出的值是否都 >= 阈值，且阈值以上的个数与参考一致
      int sel_gt = 0;
      bool ok = true;
      for (int i = 0; i < kc; ++i) {
        if (hval[(size_t)row * kc + i] < thresh - 1e-6f) ok = false;
        if (hval[(size_t)row * kc + i] > thresh + 1e-6f) sel_gt++;
      }
      int ref_gt = 0;
      for (float v : r)
        if (v > thresh + 1e-6f) ref_gt++;
      maxerr = std::max(maxerr, (double)std::abs(sel_gt - ref_gt));
      if (!ok) maxerr = 1e9;
    }
    std::printf("[topk check ] selected set vs CPU exact top-%d: %s\n", kc,
                maxerr == 0 ? "OK" : "FAIL");
    // 恢复整块随机 scores 用于性能测试
    fill_f32_kernel<<<div_up((int)(((size_t)S * S + 255) / 256), 1), 256>>>(scores,
                                                                           (size_t)S * S, 11);
    CUDA_CHECK_LAST();
  }

  if (want("topk") || want("all")) {
    constexpr int TT = 1024, NH = TT / 32;
    const size_t tk_shm = (size_t)S * 4 + (NH + 1) * RBUCK * 4;
    const bool smem_ok = tk_shm <= 220 * 1024;
    if (smem_ok) {
      CUDA_CHECK(cudaFuncSetAttribute(topk_radix_smem_kernel<TT, NH>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)tk_shm));
      auto run_smem = [&] {
        topk_radix_smem_kernel<TT, NH><<<S, TT, tk_shm>>>(scores, S, TOPK, out_idx, out_val);
      };
      run_smem();
      CUDA_CHECK_LAST();
      double t = bench_ms(run_smem, 3, 10);
      const double rd = (double)S * S * 4;  // 一遍读 + 输出
      std::printf("[topk smem  ] %8.4f ms  %8.1f GB/s(scores)  (radix-select, row in smem)\n", t,
                  to_gbps(rd, t));
    } else {
      std::printf("[topk smem  ] skipped (row %zu B > 200KB smem)\n", tk_shm);
    }

    auto run_glob = [&] {
      topk_radix_global_kernel<TT, NH><<<S, TT>>>(scores, S, TOPK, out_idx);
    };
    run_glob();
    CUDA_CHECK_LAST();
    double tg = bench_ms(run_glob, 3, 10);
    const double rdg = (double)S * S * 4;
    std::printf("[topk global] %8.4f ms  %8.1f GB/s(scores)  (5 passes streamed)\n", tg,
                to_gbps(rdg, tg));
  }

  CUDA_CHECK(cudaFree(q));
  CUDA_CHECK(cudaFree(kk));
  CUDA_CHECK(cudaFree(w));
  CUDA_CHECK(cudaFree(scores));
  CUDA_CHECK(cudaFree(out_idx));
  CUDA_CHECK(cudaFree(out_val));
  return 0;
}
