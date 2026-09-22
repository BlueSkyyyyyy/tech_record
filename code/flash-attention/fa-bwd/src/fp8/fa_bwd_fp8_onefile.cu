// =============================================================================
// fa_bwd_fp8_onefile.cu —— FlashAttention 反向（FP8）单文件实现（P3-2，golden）
// =============================================================================
// 目标：功能正确的 FP8 反向「golden」版本：真实做 E4M3/E5M2 + rowwise scaling 的
//       量化/反量化，但用 fp32 标量乘加模拟张量核的 fp32 累加（与 docs/02 设计一致）。
//       这样把「量化数值」与「MMA 调度」解耦，可作为后续 mma 版的数值基准。
//
// 设计（对齐 TE 2.14 fused_attn FP8 口径，见 docs/02-fp8-bwd-design.md）：
//   Q,K,V        -> E4M3  rowwise（每 (b,s,h) 行 over head_dim），max=448
//   dO           -> E5M2  rowwise，max=57344
//   P            -> E4M3  scale=1（P∈[0,1]）
//   dP           -> E5M2  rowwise（每 Q 行 over KV 列）
//   dS,LSE,D,acc -> fp32
//
// 四段式：
//   1) quantize_row_kernel ：fp32 q/k/v/do -> fp8 + rowwise scale（模拟 TE 在计时区外量化）
//   2) preprocess_kernel   ：反量化 Q/K 重算 LSE；反量化 dO 与 fp32 O 算 D=rowsum(dO∘O)
//   3) fa_bwd_fp8_kernel   ：1colblock，反量化载入 Q/K/V/dO，标量 fp32 累加；P/dP 量化
//   4) convert_kernel      ：fp32 累加缓冲 -> 输出（本版 fp32）
//
// 关键实现要点（逐条）：
//   * FP8 转换必须走 __nv_cvt_float_to_fp8 / __nv_cvt_fp8_to_halfraw，不能用 float(fp8)
//     （该工具链下后者返回原始位模式，见 agent_skills/kernel-opt.md 32 篇）。
//   * rowwise scale = amax / FP8_MAX，fp32 任意值；反量化 x' = float(fp8)*scale。
//   * 标量 golden 在反量化时已乘回 scale，故无需再做 sa*sb 折算（等价于真实 mma 的正确折算）。
//   * dP 用 warp shuffle 做行内 amax 得到 rowwise scale 后再量化到 E5M2。
//   * recompute P：不物化 N×N；每个 K tile 用 QKᵀ + LSE 重算。
//   * dQ 在 smem 跨 tile 累加；dK/dV 用 fp32 全局缓冲 atomicAdd。
//   * causal：整块跳过 + 对角 tile 逐元素 mask（mask 处 P=0，量化后仍为 0）。
//
// 本版本是「正确性优先」的 CUDA-core 标量实现；mma.m16n8k32 + ldmatrix + 流水留待 P3-4。
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// ----------------------------- 编译期常量 -----------------------------
static constexpr int kHeadDim = 128;
static constexpr int BM       = 64;
static constexpr int BN       = 32;
static constexpr int THREADS  = 128;
static constexpr int WM_ROWS  = BM / 4;  // 16
static constexpr int WN_ROWS  = BN / 4;  // 8

// FP8 饱和上界
static constexpr float kE4M3Max = 448.0f;
static constexpr float kE5M2Max = 57344.0f;

// 动态 smem 布局（字节）：
//   Qs[BM*HD] + Ks[BN*HD] + Vs[BN*HD] + dOs[BM*HD] + Ps[BM*BN]   （fp8, 1B）
//   qs_s[BM] + ks_s[BN] + vs_s[BN] + dos_s[BM]                    （fp32 scale）
//   Ss[BM*BN] + dQs[BM*HD]                                        （fp32）
static constexpr int kByteRegion = BM * kHeadDim + BN * kHeadDim + BN * kHeadDim +
                                   BM * kHeadDim + BM * BN;
static constexpr int kSmemBytes = kByteRegion + (BM + BN + BN + BM) * (int)sizeof(float) +
                                  (BM * BN + BM * kHeadDim) * (int)sizeof(float);

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// ----------------------------- FP8 转换 helper -----------------------------
__device__ __forceinline__ unsigned char cvt_e4m3(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ unsigned char cvt_e5m2(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2);
}
__device__ __forceinline__ float deq_e4m3(unsigned char q) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  return __half2float(__half(h));
}
__device__ __forceinline__ float deq_e5m2(unsigned char q) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E5M2);
  return __half2float(__half(h));
}

// =============================================================================
// 1) quantize_row_kernel：逐行 rowwise 量化
//   grid = (rows=B*S*H)，block = kHeadDim(=128)
//   x fp32 -> xq fp8（e4m3/e5m2）+ scale（amax/FP8_MAX）
// =============================================================================
__global__ void quantize_row_kernel(const float* __restrict__ x,
                                    unsigned char* __restrict__ xq,
                                    float* __restrict__ scale, int D, int is_e5m2) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  const float* xr = x + (size_t)row * D;
  const float v = (d < D) ? xr[d] : 0.f;

  __shared__ float sh[128];
  sh[d] = fabsf(v);
  __syncthreads();
  for (int off = 64; off > 0; off >>= 1) {
    if (d < off) sh[d] = fmaxf(sh[d], sh[d + off]);
    __syncthreads();
  }
  const float fp8_max = is_e5m2 ? kE5M2Max : kE4M3Max;
  const float s = (sh[0] > 0.f) ? (sh[0] / fp8_max) : 1.f;
  if (d == 0) scale[row] = s;
  if (d < D) {
    const float y = v / s;
    xq[(size_t)row * D + d] = is_e5m2 ? cvt_e5m2(y) : cvt_e4m3(y);
  }
}

// =============================================================================
// 2) preprocess：反量化 Q/K 求 LSE；反量化 dO 与 fp32 O 求 D
//   grid = (S, H, B)，block = THREADS
// =============================================================================
__global__ void preprocess_kernel(const unsigned char* __restrict__ q8,
                                  const float* __restrict__ qs,
                                  const unsigned char* __restrict__ k8,
                                  const float* __restrict__ ks,
                                  const float* __restrict__ o,
                                  const unsigned char* __restrict__ do8,
                                  const float* __restrict__ dos,
                                  float* __restrict__ delta, float* __restrict__ lse,
                                  int S, int H, float scale, int causal) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const size_t row = ((size_t)(b * S + s)) * H + h;
  const unsigned char* qr = q8 + row * kHeadDim;
  const float q_scale = qs[row];

  float m = -INFINITY, l = 0.f;
  const int jmax = causal ? (s + 1) : S;
  for (int j = tid; j < jmax; j += blockDim.x) {
    const size_t krow = ((size_t)(b * S + j)) * H + h;
    const unsigned char* kr = k8 + krow * kHeadDim;
    const float k_scale = ks[krow];
    float dot = 0.f;
#pragma unroll 8
    for (int d = 0; d < kHeadDim; ++d)
      dot += deq_e4m3(qr[d]) * deq_e4m3(kr[d]);
    dot *= q_scale * k_scale * scale;
    float mn = fmaxf(m, dot);
    l = l * expf(m - mn) + expf(dot - mn);
    m = mn;
  }

  __shared__ float sh_m[THREADS], sh_l[THREADS];
  sh_m[tid] = m;
  sh_l[tid] = l;
  __syncthreads();
  for (int off = THREADS / 2; off > 0; off >>= 1) {
    if (tid < off) {
      float m1 = sh_m[tid], l1 = sh_l[tid];
      float m2 = sh_m[tid + off], l2 = sh_l[tid + off];
      float mn = fmaxf(m1, m2);
      float c1 = (m1 == -INFINITY) ? 0.f : l1 * expf(m1 - mn);
      float c2 = (m2 == -INFINITY) ? 0.f : l2 * expf(m2 - mn);
      sh_m[tid] = mn;
      sh_l[tid] = c1 + c2;
    }
    __syncthreads();
  }

  // D = rowsum(O ∘ dO')，dO' 为反量化后的 dO
  float dp = 0.f;
  const float* orow = o + row * kHeadDim;
  const unsigned char* dorow = do8 + row * kHeadDim;
  const float do_scale = dos[row];
  for (int d = tid; d < kHeadDim; d += blockDim.x)
    dp += orow[d] * (deq_e5m2(dorow[d]) * do_scale);
  __shared__ float sh_delta[THREADS];
  sh_delta[tid] = dp;
  __syncthreads();
  for (int off = THREADS / 2; off > 0; off >>= 1) {
    if (tid < off) sh_delta[tid] += sh_delta[tid + off];
    __syncthreads();
  }

  if (tid == 0) {
    delta[row] = sh_delta[0];
    lse[row] = sh_m[0] + logf(sh_l[0]);
  }
}

// =============================================================================
// 3) main kernel：1colblock 反向（FP8 量化 + fp32 累加）
//   grid = (ceil(S/BM), H, B)，block = THREADS
// =============================================================================
__global__ void __launch_bounds__(THREADS)
fa_bwd_fp8_kernel(const unsigned char* __restrict__ q8,
                  const float* __restrict__ qs,
                  const unsigned char* __restrict__ k8,
                  const float* __restrict__ ks,
                  const unsigned char* __restrict__ v8,
                  const float* __restrict__ vs,
                  const unsigned char* __restrict__ do8,
                  const float* __restrict__ dos,
                  const float* __restrict__ delta,
                  const float* __restrict__ lse,
                  float* __restrict__ dq_acc,
                  float* __restrict__ dk_acc,
                  float* __restrict__ dv_acc,
                  int S, int H, float scale, int causal) {
  extern __shared__ char smem[];
  unsigned char* Qs  = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Ks  = Qs + BM * kHeadDim;
  unsigned char* Vs  = Ks + BN * kHeadDim;
  unsigned char* dOs = Vs + BN * kHeadDim;
  unsigned char* Ps  = dOs + BM * kHeadDim;
  float* scales = reinterpret_cast<float*>(Ps + BM * BN);
  float* qs_s = scales;
  float* ks_s = qs_s + BM;
  float* vs_s = ks_s + BN;
  float* dos_s = vs_s + BN;
  float* Ss = dos_s + BM;                 // dS（fp32）
  float* dQs = Ss + BM * BN;              // dQ 累加（fp32）

  const int mblk = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int m0 = mblk * BM;

  // ---- 载入 Q/dO 分块（fp8 + rowwise scale，越界补 0）----
  for (int i = tid; i < BM * kHeadDim; i += THREADS) {
    int r = i / kHeadDim, d = i % kHeadDim;
    int qi = m0 + r;
    unsigned char qv = cvt_e4m3(0.f), ov = cvt_e5m2(0.f);
    if (qi < S) {
      size_t idx = (((size_t)(b * S + qi)) * H + h) * kHeadDim + d;
      qv = q8[idx];
      ov = do8[idx];
    }
    Qs[i] = qv;
    dOs[i] = ov;
  }
  if (tid < BM) {
    int qi = m0 + tid;
    qs_s[tid] = (qi < S) ? qs[((size_t)(b * S + qi)) * H + h] : 1.f;
    dos_s[tid] = (qi < S) ? dos[((size_t)(b * S + qi)) * H + h] : 1.f;
  }
  for (int i = tid; i < BM * kHeadDim; i += THREADS) dQs[i] = 0.f;
  __syncthreads();

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;

    // ---- 载入 K/V 分块（fp8 + scale）----
    for (int i = tid; i < BN * kHeadDim; i += THREADS) {
      int j = i / kHeadDim, d = i % kHeadDim;
      int jg = j0 + j;
      unsigned char kv = cvt_e4m3(0.f), vv = cvt_e4m3(0.f);
      if (jg < S) {
        size_t idx = (((size_t)(b * S + jg)) * H + h) * kHeadDim + d;
        kv = k8[idx];
        vv = v8[idx];
      }
      Ks[i] = kv;
      Vs[i] = vv;
    }
    if (tid < BN) {
      int jg = j0 + tid;
      ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * H + h] : 1.f;
      vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * H + h] : 1.f;
    }
    __syncthreads();

    // ---- S = scale·Q'K'ᵀ；P = exp(S − LSE)，量化到 E4M3(scale=1) ----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      float dot = 0.f;
      if (qi < S) {
        const unsigned char* qrow = Qs + r * kHeadDim;
        const unsigned char* krow = Ks + lane * kHeadDim;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d)
          dot += deq_e4m3(qrow[d]) * deq_e4m3(krow[d]);
        dot *= scale * qs_s[r] * ks_s[lane];
      }
      int jg = j0 + lane;
      float p = 0.f;
      if (qi < S && jg < S && !(causal && jg > qi))
        p = expf(dot - lse[((size_t)(b * S + qi)) * H + h]);
      Ps[r * BN + lane] = cvt_e4m3(p);   // P 量化（scale=1）
    }
    __syncthreads();

    // ---- dP = dO'·V'ᵀ；rowwise 量化到 E5M2；dS = P'∘(dP−D) ----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      // 本 lane 的 dP[r, lane]（lane 即 KV 列 j 方向）
      float dpv = 0.f;
      if (qi < S) {
        const unsigned char* drow = dOs + r * kHeadDim;
        const unsigned char* vrow = Vs + lane * kHeadDim;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d)
          dpv += deq_e5m2(drow[d]) * deq_e4m3(vrow[d]);
        dpv *= dos_s[r] * vs_s[lane];
      }
      // 行内 amax（跨 lane 归约）-> rowwise scale -> E5M2 量化 -> 反量化
      float amax = fabsf(dpv);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
      const float ds_scale = (amax > 0.f) ? (amax / kE5M2Max) : 1.f;
      const float dp_q =
          (amax > 0.f) ? (deq_e5m2(cvt_e5m2(dpv / ds_scale)) * ds_scale) : dpv;
      float del = (qi < S) ? delta[((size_t)(b * S + qi)) * H + h] : 0.f;
      Ss[r * BN + lane] = deq_e4m3(Ps[r * BN + lane]) * (dp_q - del);
    }
    __syncthreads();

    // ---- dV = P'ᵀ dO'（P' E4M3, dO' E5M2）----
#pragma unroll
    for (int jj = 0; jj < WN_ROWS; ++jj) {
      int j = warp * WN_ROWS + jj;
      int jg = j0 + j;
      if (jg >= S) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int i = 0; i < BM; ++i) {
        float p = deq_e4m3(Ps[i * BN + j]);
        if (p == 0.f) continue;
        float dscale = dos_s[i];
        const unsigned char* drow = dOs + i * kHeadDim;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += p * (deq_e5m2(drow[d]) * dscale);
        }
      }
      float* base = dv_acc + (((size_t)(b * S + jg)) * H + h) * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) atomicAdd(base + lane + 32 * kk, acc[kk]);
    }

    // ---- dK = scale·dSᵀ Q'（dS fp32, Q' E4M3）----
#pragma unroll
    for (int jj = 0; jj < WN_ROWS; ++jj) {
      int j = warp * WN_ROWS + jj;
      int jg = j0 + j;
      if (jg >= S) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int i = 0; i < BM; ++i) {
        float ds = Ss[i * BN + j];
        if (ds == 0.f) continue;
        float qscale = qs_s[i];
        const unsigned char* qrow = Qs + i * kHeadDim;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += ds * (deq_e4m3(qrow[d]) * qscale);
        }
      }
      float* base = dk_acc + (((size_t)(b * S + jg)) * H + h) * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        atomicAdd(base + lane + 32 * kk, acc[kk] * scale);
    }

    // ---- dQ += scale·dS K'（dS fp32, K' E4M3），累加到 smem dQs ----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      if (qi >= S) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int j = 0; j < BN; ++j) {
        float ds = Ss[r * BN + j];
        if (ds == 0.f) continue;
        float kscale = ks_s[j];
        const unsigned char* krow = Ks + j * kHeadDim;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += ds * (deq_e4m3(krow[d]) * kscale);
        }
      }
      float* dqr = dQs + r * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        dqr[lane + 32 * kk] += acc[kk] * scale;
    }
    __syncthreads();
  }

  for (int i = tid; i < BM * kHeadDim; i += THREADS) {
    int r = i / kHeadDim;
    int qi = m0 + r;
    if (qi < S)
      dq_acc[(((size_t)(b * S + qi)) * H + h) * kHeadDim + (i % kHeadDim)] = dQs[i];
  }
}

// =============================================================================
// 4) convert：fp32 累加缓冲 -> 输出（本版直接 fp32 拷贝，便于与 ref 逐元素比对）
// =============================================================================
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc,
                               float* __restrict__ dq, float* __restrict__ dk,
                               float* __restrict__ dv, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    dq[i] = dq_acc[i];
    dk[i] = dk_acc[i];
    dv[i] = dv_acc[i];
  }
}

// =============================================================================
// 极简 npy 读取（little-endian C-contiguous float32）
// =============================================================================
struct NpyF32 {
  std::vector<float> data;
  std::vector<long> shape;
};

static NpyF32 load_npy_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "无法打开 %s\n", path.c_str());
    std::exit(1);
  }
  char magic[6];
  f.read(magic, 6);
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h16 = 0;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hlen = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }
  std::string header(hlen, '\0');
  f.read(&header[0], hlen);
  if (header.find("f4") == std::string::npos) {
    fprintf(stderr, "%s 不是 float32 npy\n", path.c_str());
    std::exit(1);
  }
  NpyF32 out;
  size_t sp = header.find("'shape'");
  size_t lp = header.find('(', sp);
  size_t rp = header.find(')', lp);
  std::string tup = header.substr(lp + 1, rp - lp - 1);
  for (size_t i = 0; i < tup.size();) {
    while (i < tup.size() && (tup[i] == ' ' || tup[i] == ',')) ++i;
    long v = 0;
    bool any = false;
    while (i < tup.size() && tup[i] >= '0' && tup[i] <= '9') {
      v = v * 10 + (tup[i] - '0');
      ++i;
      any = true;
    }
    if (any) out.shape.push_back(v);
  }
  std::streampos pos = f.tellg();
  f.seekg(0, std::ios::end);
  size_t total = (size_t)f.tellg();
  f.seekg(pos);
  size_t nbytes = total - (size_t)pos;
  out.data.resize(nbytes / sizeof(float));
  f.read(reinterpret_cast<char*>(out.data.data()), nbytes);
  return out;
}

struct DiffStat {
  double max_abs;
  double max_rel;
};

static DiffStat diff_stat(const std::vector<float>& a, const std::vector<float>& b) {
  DiffStat st{0.0, 0.0};
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    if (d > st.max_abs) st.max_abs = d;
    double r = d / (std::fabs((double)b[i]) + 1e-3);
    if (r > st.max_rel) st.max_rel = r;
  }
  return st;
}

// =============================================================================
// host / launcher / self-test
// =============================================================================
int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8";
  std::string o_name = "ref_o";
  bool causal = true;
  int iters = 20;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--full") causal = false;
    else if (a == "--causal") causal = true;
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  auto q_np  = load_npy_f32(dir + "/q.npy");
  auto k_np  = load_npy_f32(dir + "/k.npy");
  auto v_np  = load_npy_f32(dir + "/v.npy");
  auto do_np = load_npy_f32(dir + "/do.npy");
  auto o_np  = load_npy_f32(dir + "/" + o_name + ".npy");
  auto rdq   = load_npy_f32(dir + "/ref_dq.npy");
  auto rdk   = load_npy_f32(dir + "/ref_dk.npy");
  auto rdv   = load_npy_f32(dir + "/ref_dv.npy");

  if (q_np.shape.size() != 4) {
    fprintf(stderr, "期望 q 为 4D [B,S,H,D]\n");
    return 1;
  }
  const int B = (int)q_np.shape[0], S = (int)q_np.shape[1];
  const int H = (int)q_np.shape[2], D = (int)q_np.shape[3];
  if (D != kHeadDim) {
    fprintf(stderr, "本版本仅支持 head_dim=%d（当前 %d）\n", kHeadDim, D);
    return 1;
  }
  const size_t n = (size_t)B * S * H * D;
  const size_t rows = (size_t)B * S * H;
  const float scale = 1.0f / sqrtf((float)D);

  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d D=%d causal=%d scale=%.6f\n", B, S, H, D, (int)causal, scale);
  printf("FP8: Q/K/V=E4M3 rowwise, dO/dP=E5M2 rowwise, P=E4M3(scale=1), dS/acc=fp32\n");

  // device 缓冲
  float* d_q_f; float* d_k_f; float* d_v_f; float* d_do_f; float* d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc, *d_dq, *d_dk, *d_dv;
  CUDA_CHECK(cudaMalloc(&d_q_f, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k_f, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v_f, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_do_f, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o_f, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q8, n));
  CUDA_CHECK(cudaMalloc(&d_k8, n));
  CUDA_CHECK(cudaMalloc(&d_v8, n));
  CUDA_CHECK(cudaMalloc(&d_do8, n));
  CUDA_CHECK(cudaMalloc(&d_qs, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ks, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_vs, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dos, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_delta, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dk, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dv, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  // 量化（模拟 TE「计时区外预量化」）：Q/K/V E4M3，dO E5M2
  auto quant = [&]() {
    quantize_row_kernel<<<(int)rows, kHeadDim>>>(d_q_f, d_q8, d_qs, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, kHeadDim>>>(d_k_f, d_k8, d_ks, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, kHeadDim>>>(d_v_f, d_v8, d_vs, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, kHeadDim>>>(d_do_f, d_do8, d_dos, kHeadDim, 1);
  };

  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));

  dim3 pg(S, H, B);
  dim3 mg((S + BM - 1) / BM, H, B);
  const int cvt_threads = 256;
  const int cvt_blocks = (int)std::min<size_t>((n + cvt_threads - 1) / cvt_threads, 65535);

  auto run_all = [&]() {
    quant();
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, n * sizeof(float)));
    preprocess_kernel<<<pg, THREADS>>>(d_q8, d_qs, d_k8, d_ks, d_o_f, d_do8, d_dos,
                                       d_delta, d_lse, S, H, scale, (int)causal);
    fa_bwd_fp8_kernel<<<mg, THREADS, kSmemBytes>>>(d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                   d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, S, H, scale,
                                                   (int)causal);
    convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                d_dv, n);
  };

  for (int i = 0; i < 3; ++i) run_all();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_all();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  ms /= iters;
  double flops = 4.0 * (double)B * S * H * S * D;
  printf("[timing] total(quant+pre+main+cvt) %.4f ms  %.2f TFLOPS (bwd FLOPs=4BS^2HD)\n",
         ms, flops / (ms * 1e-3) / 1e12);

  // 分段
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) quant();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_quant = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_quant, ev0, ev1));
  ms_quant /= iters;

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i)
    preprocess_kernel<<<pg, THREADS>>>(d_q8, d_qs, d_k8, d_ks, d_o_f, d_do8, d_dos,
                                       d_delta, d_lse, S, H, scale, (int)causal);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_pre = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev0, ev1));
  ms_pre /= iters;

  CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dk_acc, 0, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dv_acc, 0, n * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i)
    fa_bwd_fp8_kernel<<<mg, THREADS, kSmemBytes>>>(d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                   d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, S, H, scale,
                                                   (int)causal);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_main = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_main, ev0, ev1));
  ms_main /= iters;
  printf("[timing] quant %.4f ms | preprocess %.4f ms | main %.4f ms | convert %.4f ms\n",
         ms_quant, ms_pre, ms_main, ms - ms_quant - ms_pre - ms_main);

  // 数值对拍
  std::vector<float> mdq(n), mdk(n), mdv(n);
  CUDA_CHECK(cudaMemcpy(mdq.data(), d_dq, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdk.data(), d_dk, n * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdv.data(), d_dv, n * sizeof(float), cudaMemcpyDeviceToHost));

  auto print_cmp = [&](const char* name, const std::vector<float>& mine,
                       const std::vector<float>& ref) {
    DiffStat st = diff_stat(mine, ref);
    printf("  %-4s vs ref: max_abs=%.3e  max_rel=%.3e\n", name, st.max_abs, st.max_rel);
  };
  printf("[compare] ours vs fp32 ref (O from %s.npy)\n", o_name.c_str());
  print_cmp("dq", mdq, rdq.data);
  print_cmp("dk", mdk, rdk.data);
  print_cmp("dv", mdv, rdv.data);

  auto try_cmp = [&](const char* fname, const std::vector<float>& ref) {
    std::ifstream test(dir + "/" + fname + ".npy");
    if (!test.good()) return;
    auto t = load_npy_f32(dir + "/" + fname + ".npy");
    DiffStat st = diff_stat(t.data, ref);
    printf("  %-9s vs ref: max_abs=%.3e  max_rel=%.3e\n", fname, st.max_abs, st.max_rel);
  };
  try_cmp("te_dq", rdq.data);
  try_cmp("te_dk", rdk.data);
  try_cmp("te_dv", rdv.data);

  // ours vs TE FP8：同为 FP8 口径，比 ours-vs-ref 更能反映实现差异
  auto cmp_to = [&](const char* name, const std::vector<float>& mine,
                    const char* fname) {
    std::ifstream test(dir + "/" + fname + ".npy");
    if (!test.good()) return;
    auto t = load_npy_f32(dir + "/" + fname + ".npy");
    DiffStat st = diff_stat(mine, t.data);
    printf("  %-4s vs %s: max_abs=%.3e  max_rel=%.3e\n", name, fname, st.max_abs,
           st.max_rel);
  };
  printf("[compare] ours vs TE FP8\n");
  cmp_to("dq", mdq, "te_dq");
  cmp_to("dk", mdk, "te_dk");
  cmp_to("dv", mdv, "te_dv");

  cudaFree(d_q_f); cudaFree(d_k_f); cudaFree(d_v_f); cudaFree(d_do_f); cudaFree(d_o_f);
  cudaFree(d_q8); cudaFree(d_k8); cudaFree(d_v8); cudaFree(d_do8);
  cudaFree(d_qs); cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_dos);
  cudaFree(d_delta); cudaFree(d_lse);
  cudaFree(d_dq_acc); cudaFree(d_dk_acc); cudaFree(d_dv_acc);
  cudaFree(d_dq); cudaFree(d_dk); cudaFree(d_dv);
  return 0;
}
