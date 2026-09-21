// 35 DSA compressor：把 KV 按 compress_ratio 做「门控池化」压成更少的 key
//
// 真实算子来自 DeepSeek-V4 官方推理实现
//   /ssd/models/DeepSeek-V4-Pro/inference/model.py:279（class Compressor）
// DSA 稀疏注意力只在「压缩后的 KV」上算 indexer / top-k；compressor 就是那张
// 「把 r 个连续 token 的 KV 用学习到的门控 softmax 池化成一个 compressed key」的算子。
//
// 参考实现的 prefill（start_pos==0）语义：
//   kv    = x @ wkv^T      # [S, coff*d]   coff = 1 + (ratio==4)
//   score = x @ wgate^T    # [S, coff*d]
//   按 ratio 个 token 一组：score[win, i, :] += ape[i, :]
//   out[win, :] = sum_i softmax_i(score) * kv        # 每个输出列独立做一次 softmax
//   若 ratio==4（overlap）：每个压缩 token 用 2*ratio 个 token 的「重叠窗」——
//     前半 slots 来自上一个窗的前一半 dim、后半 slots 来自本窗的后一半 dim。
//   最后 out = RMSNorm(out)，对末 rope_head_dim(64) 维做 RoPE，再写回 kv_cache。
//
// 真实 shape（/ssd/models/DeepSeek-V4-Pro/config.json）：
//   hidden_size=7168, head_dim=512, qk_rope_head_dim=64,
//   compress_ratios=[128,...,4,...]（ratio 128 与 4 交替，末层 0=不压缩）
//   投影权重 wkv/wgate 在 checkpoint 里是 bf16（参考实现用 fp32 计算）。
//
// 本文件实现：
//   1) proj_ws_kernel : 把 wkv 与 wgate 合成一个 [2C, D] 的 GEMM（A 只读一遍），
//                       复用 23/28 篇的 TMA + mbarrier + warp specialization + wgmma SW128，
//                       输出 Y[S, 2C] fp32（[:, :C]=kv，[:, C:]=score）。
//   2) pool_kernel    : 逐窗 online-softmax 门控池化 + RMSNorm + RoPE（融合），输出 [S/r, d] bf16。
//
// 运行：
//   ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//     scripts/run.sh 35-dsa-compressor/compressor.cu [ratio] [M] [which]
#include "../common/cuda_utils.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;
constexpr double BF16_PEAK = 989.0;

// DeepSeek-V4-Pro compressor shape
constexpr int D  = 7168;  // hidden_size (K)
constexpr int HD = 512;   // head_dim (compressed dim d)
constexpr int RD = 64;    // qk_rope_head_dim

// ---------------------------------------------------------------------------
// wgmma / SW128 helpers（同 20/28 篇）
// ---------------------------------------------------------------------------
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ int sw128_off(int row, int k) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ uint32_t k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}

// ---------------------------------------------------------------------------
// mbarrier + TMA helpers（同 23/28）
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred p;\n"
      "WAIT_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra WAIT_%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(phase));
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void fence_mbar_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, void* dst, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}

// ---------------------------------------------------------------------------
// (1) 合并投影 GEMM：Y[M,N] = X[M,K] @ Wm[N,K]^T
//     TMA + mbarrier + warp specialization + wgmma SS SW128（N 为 2C）
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, bool BF16OUT = false>
__global__ void __launch_bounds__((BM / 64) * 128 + 32)
proj_ws_kernel(const __grid_constant__ CUtensorMap tmA,
               const __grid_constant__ CUtensorMap tmB,
               void* __restrict__ C, int M, int N, int K) {
  static_assert(BK == 64, "bf16 SW128 内维固定 64 元素");
  constexpr int NWG = BM / 64;
  constexpr int NSPLIT = BN / 128;
  constexpr int NCONS = NWG * 128;
  constexpr int SBO = 1024;

  constexpr int BARR = ((int)(2 * STAGES * sizeof(uint64_t)) + 1023) / 1024 * 1024;
  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + BARR;
  char* Bs = As + (size_t)STAGES * BM * 128;

  const int tid = threadIdx.x;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) { mbar_init(full + s, 1); mbar_init(empty + s, NCONS); }
    fence_mbar_init();
  }
  __syncthreads();

  if (tid >= NCONS) {
    if (tid == NCONS) {
      for (int q = 0; q < nblk; ++q) {
        const int st = q % STAGES;
        if (q >= STAGES) mbar_wait(empty + st, (uint32_t)((q / STAGES - 1) & 1));
        fence_proxy_async();
        mbar_arrive_expect_tx(full + st, BM * 128 + BN * 128);
        tma_load_2d(&tmA, As + (size_t)st * BM * 128, q * 128, block_row, full + st);
        tma_load_2d(&tmB, Bs + (size_t)st * BN * 128, q * 128, block_col, full + st);
      }
    }
    return;
  }

  const int wg = tid >> 7;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float acc[NSPLIT][64];
#pragma unroll
  for (int j = 0; j < NSPLIT; ++j)
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[j][i] = 0.f;

  for (int q = 0; q < nblk; ++q) {
    const int st = q % STAGES;
    mbar_wait(full + st, (uint32_t)((q / STAGES) & 1));
    fence_proxy_async();
    char* a = As + (size_t)st * BM * 128 + (size_t)wg * 64 * 128;
    char* b = Bs + (size_t)st * BN * 128;
    wgmma_fence();
#pragma unroll
    for (int jn = 0; jn < NSPLIT; ++jn) {
      char* bj = b + (size_t)jn * 128 * 128;
#pragma unroll
      for (int s = 0; s < BK / 16; ++s) {
        uint64_t da = make_desc_sw128(k16_addr(smem_u32(a), s), SBO);
        uint64_t db = make_desc_sw128(k16_addr(smem_u32(bj), s), SBO);
        wgmma_m64n128k16(acc[jn], da, db);
      }
    }
    wgmma_commit();
    if (q >= STAGES - 2) {
      wgmma_wait_group<STAGES - 2>();
      const int rs = (q - (STAGES - 2)) % STAGES;
      mbar_arrive(empty + rs);
    }
  }
  wgmma_wait0();

  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
  float* Cf = reinterpret_cast<float*>(C);
  bf16* Cb = reinterpret_cast<bf16*>(C);
#pragma unroll
  for (int jn = 0; jn < NSPLIT; ++jn)
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = jn * 128 + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8, cc = block_col + col;
      if constexpr (BF16OUT) {
        if (r0 < M) *reinterpret_cast<__nv_bfloat162*>(&Cb[(size_t)r0 * N + cc]) =
            __floats2bfloat162_rn(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<__nv_bfloat162*>(&Cb[(size_t)r1 * N + cc]) =
            __floats2bfloat162_rn(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
      } else {
        if (r0 < M) *reinterpret_cast<float2*>(&Cf[(size_t)r0 * N + cc]) = make_float2(acc[jn][j * 4 + 0], acc[jn][j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<float2*>(&Cf[(size_t)r1 * N + cc]) = make_float2(acc[jn][j * 4 + 2], acc[jn][j * 4 + 3]);
      }
    }
}

// ---------------------------------------------------------------------------
// (2) 门控池化 + RMSNorm + RoPE，融合成一个 kernel
//     一个 block = 一个输出窗（ratio 或 2*ratio 个 slot），线程 j = 列 j∈[0,HD)
//     RATIO=128: coff=1，128 个 slot（本窗）
//     RATIO=4  : coff=2，overlap，8 个 slot（上窗前一半 + 本窗后一半）
// ---------------------------------------------------------------------------
template <int RATIO, bool BF16IN = false>
__global__ void __launch_bounds__(HD)
pool_kernel(const void* __restrict__ Y,      // [M, 2C]
            const float* __restrict__ ape,   // [RATIO, C]
            const float* __restrict__ nw,    // [HD]
            const float* __restrict__ cos_t, // [M/RATIO, RD/2]
            const float* __restrict__ sin_t,
            bf16* __restrict__ out, int M, int N, int C) {
  constexpr int COFF = (RATIO == 4) ? 2 : 1;
  const int t = blockIdx.x;
  const int j = threadIdx.x;
  const int tok0 = t * RATIO;
  const float* Yf = reinterpret_cast<const float*>(Y);
  const bf16* Yb = reinterpret_cast<const bf16*>(Y);
  auto rdy = [&](size_t i) { return BF16IN ? __bfloat162float(Yb[i]) : Yf[i]; };

  float m = -1e30f, l = 0.f, acc = 0.f;
  auto add_slot = [&](float kv_v, float sc_v) {
    const float mn = fmaxf(m, sc_v);
    const float a = __expf(m - mn);
    const float p = __expf(sc_v - mn);
    l = l * a + p;
    acc = acc * a + p * kv_v;
    m = mn;
  };

  if constexpr (RATIO == 128) {
#pragma unroll 4
    for (int i = 0; i < RATIO; ++i) {
      const size_t base = (size_t)(tok0 + i) * N;
      const float sc = rdy(base + C + j) + ape[i * C + j];
      add_slot(rdy(base + j), sc);
    }
  } else {
    // overlap：slots 0..RATIO-1 = 上一窗（token (t-1)*RATIO+i）的前半 dim
    //           slots RATIO..2RATIO-1 = 本窗（token t*RATIO+i）的后半 dim
    if (t > 0) {
#pragma unroll
      for (int i = 0; i < RATIO; ++i) {
        const size_t base = (size_t)(tok0 - RATIO + i) * N;
        const float sc = rdy(base + C + j) + ape[i * C + j];
        add_slot(rdy(base + j), sc);
      }
    }
#pragma unroll
    for (int i = 0; i < RATIO; ++i) {
      const size_t base = (size_t)(tok0 + i) * N;
      const float sc = rdy(base + C + HD + j) + ape[i * C + HD + j];
      add_slot(rdy(base + HD + j), sc);
    }
  }
  const float v0 = acc / l;

  // ---- RMSNorm（窗内 HD 列）----
  __shared__ float sred[HD / 32];
  float ss = v0 * v0;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, o);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) sred[warp] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float s = (threadIdx.x < HD / 32) ? sred[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (threadIdx.x == 0) sred[0] = s;
  }
  __syncthreads();
  const float rstd = rsqrtf(sred[0] / HD + 1e-6f);
  float v = v0 * rstd * nw[j];

  // ---- RoPE（末 RD 维），需要成对 → 经 smem 交换 ----
  __shared__ float srow[HD];
  srow[j] = v;
  __syncthreads();
  if (j >= HD - RD) {
    const int p = (j - (HD - RD)) >> 1;       // pair index 0..RD/2-1
    const int par = (j - (HD - RD)) & 1;
    const float a0 = srow[HD - RD + 2 * p];
    const float a1 = srow[HD - RD + 2 * p + 1];
    const int ft = t * (RD / 2) + p;
    const float c = cos_t[ft], s = sin_t[ft];
    v = par == 0 ? (a0 * c - a1 * s) : (a0 * s + a1 * c);
  }
  out[(size_t)t * HD + j] = __float2bfloat16(v);
}

// ---------------------------------------------------------------------------
// host：freqs（yarn）与 CPU 参考
// ---------------------------------------------------------------------------
static double yarn_freq(int m, int rope_dim, double base, double factor,
                        double original_seq_len, int beta_fast, int beta_slow) {
  auto fdim = [&](double num_rot) {
    return rope_dim * std::log(original_seq_len / (num_rot * 2 * M_PI)) / (2 * std::log(base));
  };
  double lo = std::floor(fdim(beta_fast)), hi = std::ceil(fdim(beta_slow));
  lo = std::max(lo, 0.0); hi = std::min(hi, (double)rope_dim - 1);
  double low = lo, high = hi;
  if (low == high) high += 0.001;
  double ramp = (m - low) / (high - low);
  ramp = std::min(1.0, std::max(0.0, ramp));
  double smooth = 1.0 - ramp;
  double f = 1.0 / std::pow(base, (double)(2 * m) / rope_dim);
  return f / factor * (1.0 - smooth) + f * smooth;
}

// 完整 CPU 参考：x[M,D] bf16(as float), Wm[N,D], ape, nw, 输出 [M/RATIO, HD]
static void cpu_ref(const std::vector<float>& x, const std::vector<float>& Wm,
                    const std::vector<float>& ape, const std::vector<float>& nw,
                    int M, int N, int C, int RATIO, std::vector<float>& out) {
  const int W = M / RATIO;
  out.assign((size_t)W * HD, 0.f);
  std::vector<double> ycol(M);  // scratch per column
  std::vector<float> srow(HD);
  auto freq = [&](int m) {
    return yarn_freq(m, RD, 160000.0, 16.0, 65536.0, 32, 1);
  };
  for (int t = 0; t < W; ++t) {
    for (int j = 0; j < HD; ++j) {
      // 收集 slots
      float m = -1e30f, l = 0.f, acc = 0.f;
      auto add = [&](double kv, double sc) {
        double mn = std::max((double)m, sc);
        double a = std::exp((double)m - mn);
        double p = std::exp(sc - mn);
        l = (float)(l * a + p);
        acc = (float)(acc * a + p * kv);
        m = (float)mn;
      };
      if (RATIO == 128) {
        for (int i = 0; i < RATIO; ++i) {
          int tok = t * RATIO + i;
          double kv = 0, sc = 0;
          for (int k = 0; k < D; ++k) { kv += (double)x[(size_t)tok * D + k] * Wm[(size_t)j * D + k]; }
          for (int k = 0; k < D; ++k) { sc += (double)x[(size_t)tok * D + k] * Wm[(size_t)(C + j) * D + k]; }
          sc += ape[(size_t)i * C + j];
          add(kv, sc);
        }
      } else {
        if (t > 0) for (int i = 0; i < RATIO; ++i) {
          int tok = (t - 1) * RATIO + i;
          double kv = 0, sc = 0;
          for (int k = 0; k < D; ++k) { kv += (double)x[(size_t)tok * D + k] * Wm[(size_t)j * D + k]; }
          for (int k = 0; k < D; ++k) { sc += (double)x[(size_t)tok * D + k] * Wm[(size_t)(C + j) * D + k]; }
          sc += ape[(size_t)i * C + j];
          add(kv, sc);
        }
        for (int i = 0; i < RATIO; ++i) {
          int tok = t * RATIO + i;
          double kv = 0, sc = 0;
          for (int k = 0; k < D; ++k) { kv += (double)x[(size_t)tok * D + k] * Wm[(size_t)(HD + j) * D + k]; }
          for (int k = 0; k < D; ++k) { sc += (double)x[(size_t)tok * D + k] * Wm[(size_t)(C + HD + j) * D + k]; }
          sc += ape[(size_t)i * C + HD + j];
          add(kv, sc);
        }
      }
      srow[j] = acc / l;
    }
    double ss = 0;
    for (int j = 0; j < HD; ++j) ss += (double)srow[j] * srow[j];
    float rstd = (float)(1.0 / std::sqrt(ss / HD + 1e-6));
    for (int j = 0; j < HD; ++j) srow[j] = srow[j] * rstd * nw[j];
    // rope
    for (int p = 0; p < RD / 2; ++p) {
      float a0 = srow[HD - RD + 2 * p], a1 = srow[HD - RD + 2 * p + 1];
      double ang = (double)t * freq(p);
      float c = (float)std::cos(ang), s = (float)std::sin(ang);
      srow[HD - RD + 2 * p] = a0 * c - a1 * s;
      srow[HD - RD + 2 * p + 1] = a0 * s + a1 * c;
    }
    for (int j = 0; j < HD; ++j) out[(size_t)t * HD + j] = srow[j];
  }
}

// ---------------------------------------------------------------------------
static CUtensorMap make_tmap(const void* ptr, uint64_t rowbytes, uint64_t rows, uint32_t boxR) {
  CUtensorMap tm;
  cuuint64_t dims[2] = {rowbytes, rows};
  cuuint64_t strides[1] = {rowbytes};
  cuuint32_t box[2] = {128, boxR};
  cuuint32_t es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr), dims, strides, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    std::fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", s ? s : "?");
    std::exit(1);
  }
  return tm;
}

int main(int argc, char** argv) {
  int RATIO = (argc > 1) ? std::atoi(argv[1]) : 128;
  int M     = (argc > 2) ? std::atoi(argv[2]) : 32768;
  const char* which = (argc > 3) ? argv[3] : "all";
  const bool skipcorr = (argc > 4) ? (std::atoi(argv[4]) != 0) : false;
  if (RATIO != 128 && RATIO != 4) { std::fprintf(stderr, "ratio must be 128 or 4\n"); return 1; }
  const int COFF = (RATIO == 4) ? 2 : 1;
  const int C = COFF * HD;
  const int N = 2 * C;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * D;
  std::printf("\nDSA Compressor (ratio=%d, coff=%d): M=%d  Y[M,%d]=X[M,%d]@Wm[%d,%d]^T\n",
              RATIO, COFF, M, N, D, N, D);
  std::printf("proj FLOPs=%.2f GFLOP  out=[%d, %d]\n\n", flops / 1e9, M / RATIO, HD);

  // ---- 构造数据 ----
  std::vector<bf16> hX((size_t)M * D), hWm((size_t)N * D);
  std::vector<float> hApe((size_t)RATIO * C), hNw(HD);
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (auto& v : hX) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hWm) v = __float2bfloat16(0.05f * nd(rng));
  for (auto& v : hApe) v = 0.25f * nd(rng);
  for (auto& v : hNw) v = 1.f + 0.1f * nd(rng);

  bf16 *dX, *dWm, *dOut;
  float *dY, *dApe, *dNw;
  CUDA_CHECK(cudaMalloc(&dX, (size_t)M * D * 2));
  CUDA_CHECK(cudaMalloc(&dWm, (size_t)N * D * 2));
  CUDA_CHECK(cudaMalloc(&dY, (size_t)M * N * 4));
  CUDA_CHECK(cudaMalloc(&dApe, (size_t)RATIO * C * 4));
  CUDA_CHECK(cudaMalloc(&dNw, HD * 4));
  CUDA_CHECK(cudaMalloc(&dOut, (size_t)(M / RATIO) * HD * 2));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), (size_t)M * D * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWm, hWm.data(), (size_t)N * D * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dApe, hApe.data(), (size_t)RATIO * C * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dNw, hNw.data(), HD * 4, cudaMemcpyHostToDevice));

  // cos/sin
  const int NWIN = M / RATIO;
  std::vector<float> hcos((size_t)NWIN * (RD / 2)), hsin((size_t)NWIN * (RD / 2));
  for (int t = 0; t < NWIN; ++t)
    for (int p = 0; p < RD / 2; ++p) {
      double ang = (double)t * yarn_freq(p, RD, 160000.0, 16.0, 65536.0, 32, 1);
      hcos[(size_t)t * (RD / 2) + p] = (float)std::cos(ang);
      hsin[(size_t)t * (RD / 2) + p] = (float)std::sin(ang);
    }
  float *dcos, *dsin;
  CUDA_CHECK(cudaMalloc(&dcos, hcos.size() * 4));
  CUDA_CHECK(cudaMalloc(&dsin, hsin.size() * 4));
  CUDA_CHECK(cudaMemcpy(dcos, hcos.data(), hcos.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsin, hsin.data(), hsin.size() * 4, cudaMemcpyHostToDevice));

  auto want = [&](const char* name) { return std::strcmp(which, name) == 0; };
  // which 是某个具体 config 名时，只跑那一个 config（便于 ncu 单核剖析）
  const bool named_cfg = !(want("all") || want("sweep") || want("gemm") || want("pool"));

  // ---- 正确性：CPU 全流程参考（小 M）----
  if (!skipcorr) {
    const int MC = 2 * RATIO;  // 2 个输出窗
    std::vector<float> xf((size_t)MC * D), wf((size_t)N * D);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = __bfloat162float(hX[i]);
    for (size_t i = 0; i < wf.size(); ++i) wf[i] = __bfloat162float(hWm[i]);
    std::vector<float> ref;
    cpu_ref(xf, wf, hApe, hNw, MC, N, C, RATIO, ref);

    // GPU：小 M 跑 GEMM + pool
    CUtensorMap tmA = make_tmap(dX, (uint64_t)D * 2, M > 256 ? M : 256, 128);
    CUtensorMap tmB = make_tmap(dWm, (uint64_t)D * 2, N, 128);
    auto fn = proj_ws_kernel<128, 128, 64, 3>;
    const int nt = 128 * 2 + 32;
    const size_t shm = ((size_t)(2 * 3 * 8) + 1023) / 1024 * 1024 + (size_t)3 * (128 + 128) * 128;
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    dim3 g(div_up(N, 128), div_up(MC, 128));
    fn<<<g, nt, shm>>>(tmA, tmB, dY, MC, N, D);
    CUDA_CHECK_LAST();
    if (RATIO == 128) pool_kernel<128><<<MC / 128, HD>>>(dY, dApe, dNw, dcos, dsin, dOut, MC, N, C);
    else pool_kernel<4><<<MC / 4, HD>>>(dY, dApe, dNw, dcos, dsin, dOut, MC, N, C);
    CUDA_CHECK_LAST();
    std::vector<bf16> ho((size_t)(MC / RATIO) * HD);
    CUDA_CHECK(cudaMemcpy(ho.data(), dOut, ho.size() * 2, cudaMemcpyDeviceToHost));
    double err = 0, renorm = 0;
    for (size_t i = 0; i < ho.size(); ++i) {
      double a = __bfloat162float(ho[i]), b = ref[i];
      err = std::max(err, std::fabs(a - b));
      renorm = std::max(renorm, std::fabs(b));
    }
    std::printf("correctness (M=%d): max_abs_err=%.3e (ref~%.3f) %s\n\n", MC, err, renorm,
                err / std::max(renorm, 1.0) < 3e-2 ? "OK" : "FAIL");
  }

  // ---- 性能：GEMM + pool ----
  auto run_gemm = [&]() {
    CUtensorMap tmA = make_tmap(dX, (uint64_t)D * 2, M, 256);
    CUtensorMap tmB = make_tmap(dWm, (uint64_t)D * 2, N, 128);
    auto fn = proj_ws_kernel<256, 128, 64, 4>;
    const int nt = 256 / 64 * 128 + 32;
    const size_t shm = ((size_t)(2 * 4 * 8) + 1023) / 1024 * 1024 + (size_t)4 * (256 + 128) * 128;
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    dim3 g(div_up(N, 128), div_up(M, 256));
    fn<<<g, nt, shm>>>(tmA, tmB, dY, M, N, D);
  };
  auto run_pool = [&]() {
    if (RATIO == 128) pool_kernel<128><<<NWIN, HD>>>(dY, dApe, dNw, dcos, dsin, dOut, M, N, C);
    else pool_kernel<4><<<NWIN, HD>>>(dY, dApe, dNw, dcos, dsin, dOut, M, N, C);
  };

#define RUN_PROJ(NAME, BM, BN, ST)                                                              \
  do {                                                                                          \
    if (want("sweep") || (named_cfg && want(NAME))) {                                        \
      CUtensorMap tmA = make_tmap(dX, (uint64_t)D * 2, M, BM);                                  \
      CUtensorMap tmB = make_tmap(dWm, (uint64_t)D * 2, N, BN);                                 \
      auto fn = proj_ws_kernel<BM, BN, 64, ST>;                                                 \
      const int nt = (BM / 64) * 128 + 32;                                                      \
      const int BARR = ((int)(2 * (ST) * sizeof(uint64_t)) + 1023) / 1024 * 1024;               \
      const size_t shm = (size_t)BARR + (size_t)(ST) * ((BM) + (BN)) * 128;                     \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm)); \
      dim3 g(div_up(N, BN), div_up(M, BM));                                                     \
      auto run = [&] { fn<<<g, nt, shm>>>(tmA, tmB, dY, M, N, D); };                            \
      run(); CUDA_CHECK_LAST();                                                                  \
      double t = bench_ms(run, 200, 100);                                                          \
      std::printf("  %-16s %8.4f ms  %8.2f TFLOPS  (%5.1f%%)\n", NAME, t,                       \
                  to_tflops(flops, t), 100.0 * to_tflops(flops, t) / BF16_PEAK);                \
    }                                                                                           \
  } while (0)

  {
    if (want("sweep")) std::printf("-- proj GEMM sweep (BM x BN, BK=64) --\n");
    RUN_PROJ("128x128s2", 128, 128, 2);
    RUN_PROJ("128x128s3", 128, 128, 3);
    RUN_PROJ("128x128s4", 128, 128, 4);
    RUN_PROJ("128x256s2", 128, 256, 2);
    RUN_PROJ("128x256s3", 128, 256, 3);
    RUN_PROJ("128x256s4", 128, 256, 4);
    RUN_PROJ("256x128s2", 256, 128, 2);
    RUN_PROJ("256x128s3", 256, 128, 3);
    RUN_PROJ("256x128s4", 256, 128, 4);
    RUN_PROJ("192x128s3", 192, 128, 3);
    RUN_PROJ("192x256s3", 192, 256, 3);
  }
  if (want("gemm")) {
    run_gemm();
    CUDA_CHECK_LAST();
    double t = bench_ms(run_gemm, 200, 100);
    std::printf("proj_ws(256x128x64s4)  %8.4f ms  %8.2f TFLOPS  (%5.1f%% of bf16 peak)\n",
                t, to_tflops(flops, t), 100.0 * to_tflops(flops, t) / BF16_PEAK);
  }
  if (want("pool") || want("all")) {
    run_gemm();
    run_pool();
    CUDA_CHECK_LAST();
    double tp = bench_ms(run_pool, 50, 100);
    double bytes = (double)M * N * 4;  // 只读 Y
    report_mem("pool(read Y)", tp, bytes, d);
    double tg = bench_ms(run_gemm, 200, 100);
    std::printf("proj_ws(256x128x64s4)  %8.4f ms  %8.2f TFLOPS  (%5.1f%% of bf16 peak)\n",
                tg, to_tflops(flops, tg), 100.0 * to_tflops(flops, tg) / BF16_PEAK);
    std::printf("end-to-end (proj + pool)  %8.4f ms\n", tg + tp);

    // ---- bf16 中间张量：Y 写读各减半 ----
    bf16* dYb; bf16* dOutb;
    CUDA_CHECK(cudaMalloc(&dYb, (size_t)M * N * 2));
    CUDA_CHECK(cudaMalloc(&dOutb, (size_t)NWIN * HD * 2));
    CUtensorMap tmA2 = make_tmap(dX, (uint64_t)D * 2, M, 256);
    CUtensorMap tmB2 = make_tmap(dWm, (uint64_t)D * 2, N, 128);
    auto fnb = proj_ws_kernel<256, 128, 64, 4, true>;
    const int ntb = 256 / 64 * 128 + 32;
    const size_t shmb = ((size_t)(2 * 4 * 8) + 1023) / 1024 * 1024 + (size_t)4 * (256 + 128) * 128;
    CUDA_CHECK(cudaFuncSetAttribute(fnb, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmb));
    dim3 gb(div_up(N, 128), div_up(M, 256));
    auto rgi = [&] { fnb<<<gb, ntb, shmb>>>(tmA2, tmB2, dYb, M, N, D); };
    auto rpi = [&] {
      if (RATIO == 128) pool_kernel<128, true><<<NWIN, HD>>>(dYb, dApe, dNw, dcos, dsin, dOutb, M, N, C);
      else pool_kernel<4, true><<<NWIN, HD>>>(dYb, dApe, dNw, dcos, dsin, dOutb, M, N, C);
    };
    rgi(); rpi(); CUDA_CHECK_LAST();
    std::vector<bf16> hof((size_t)NWIN * HD), hob((size_t)NWIN * HD);
    CUDA_CHECK(cudaMemcpy(hof.data(), dOut, hof.size() * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hob.data(), dOutb, hob.size() * 2, cudaMemcpyDeviceToHost));
    double diff = 0, refn = 0;
    for (size_t i = 0; i < hof.size(); ++i) {
      diff = std::max(diff, std::fabs((double)__bfloat162float(hof[i]) - __bfloat162float(hob[i])));
      refn = std::max(refn, std::fabs((double)__bfloat162float(hof[i])));
    }
    double tgb = bench_ms(rgi, 100, 100), tpb = bench_ms(rpi, 100, 100);
    std::printf("bf16-Y proj %8.4f ms (%.1f TFLOPS)  pool %8.4f ms  e2e %8.4f ms  "
                "vs fp32-Y out diff=%.3e (ref~%.2f)\n",
                tgb, to_tflops(flops, tgb), tpb, tgb + tpb, diff / std::max(refn, 1e-6), refn);
    CUDA_CHECK(cudaFree(dYb));
    CUDA_CHECK(cudaFree(dOutb));
  }

  CUDA_CHECK(cudaFree(dX)); CUDA_CHECK(cudaFree(dWm)); CUDA_CHECK(cudaFree(dY));
  CUDA_CHECK(cudaFree(dApe)); CUDA_CHECK(cudaFree(dNw)); CUDA_CHECK(cudaFree(dOut));
  CUDA_CHECK(cudaFree(dcos)); CUDA_CHECK(cudaFree(dsin));
  return 0;
}
