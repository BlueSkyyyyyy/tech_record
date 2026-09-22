// =============================================================================
// fa_bwd_fp16_onefile.cu —— FlashAttention 反向（fp16）单文件实现（P1-1）
// =============================================================================
// 目标：功能正确、结构清晰的 FA2 风格反向，作为后续 bf16/fp8 与两文件版的骨架。
//
// 数学（对因果/非因果均成立，见 docs/00 与 ref_impl.py）：
//   S  = scale · Q Kᵀ
//   P  = softmax(S)
//   D  = rowsum(dO ∘ O)                         （preprocess 预计算）
//   dP = dO Vᵀ
//   dS = P ∘ (dP − D)
//   dV = Pᵀ dO
//   dQ = scale · dS K
//   dK = scale · dSᵀ Q
//
// 三段式（对齐 FA2）：
//   1) preprocess_kernel：逐行求 LSE=logsumexp(scale·QKᵀ) 与 delta=rowsum(dO∘O)。
//      说明：FA 的 forward 会把 LSE 存下来给 backward 用；本仓库 dump 里只有 O，
//      没有一个现成的 LSE，所以这里在 preprocess 里重算（数学上等价于前向的 LSE）。
//      D 的预计算让主 kernel 不必同时驻留 O 与 dO。
//   2) fa_bwd_fp16_kernel：每个 Q 块固定，遍历 K/V 块（1colblock），
//      重算 S/P（recompute，不存 O(N²) 的 P），沿 K 块累加 dQ、沿 Q 块 atomic 累加 dK/dV。
//   3) convert_kernel：把 fp32 累加缓冲 dq/dk/dv 转回 fp16 写回。
//
// 关键实现要点（逐条）：
//   * recompute P：不物化 N×N 的 P；每个 K tile 用 QKᵀ + LSE 重算。
//   * D=rowsum(dO∘O) 预计算：主 kernel 只多读 delta 一行。
//   * dQ 在 smem 里跨 K tile 累加（同一 CTA 独占这些 Q 行，无需 atomic）。
//   * dK/dV 用 fp32 全局缓冲 atomicAdd（多个 Q 块贡献同一 K/V 行）。
//   * fp32 累加：S/P/dS/dQ/dK/dV 全部 fp32，只在 smem 里存 fp16 的 Q/K/V/dO。
//   * causal：整块跳过（j0 >= q_block_end 的 tile 不算），对角 tile 逐元素 mask。
//   * 动态 smem：96KB > 48KB 静态上限，需要 cudaFuncSetAttribute。
//
// 本版本是**正确性优先的 CUDA-core（标量）实现**；张量核/流水优化留待 P1-3 之后。
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// ----------------------------- 编译期常量 -----------------------------
static constexpr int kHeadDim = 128;   // 本版本固定 head_dim=128
static constexpr int BM       = 64;    // Q 块行数
static constexpr int BN       = 32;    // K/V 块行数
static constexpr int THREADS  = 128;   // 4 warps
static constexpr int WM_ROWS  = BM / 4;  // 每个 warp 负责的 Q 行数 = 16
static constexpr int WN_ROWS  = BN / 4;  // 每个 warp 负责的 K/V 行数 = 8

// 动态 smem 布局（字节）：
//   Qs[BM*HD] + Ks[BN*HD] + Vs[BN*HD] + dOs[BM*HD]   （half）
//   Ss[BM*BN] + Ps[BM*BN]                             （float）
//   dQs[BM*HD]                                        （float）
static constexpr int SMEM_BYTES =
    (BM * kHeadDim + BN * kHeadDim + BN * kHeadDim + BM * kHeadDim) * (int)sizeof(__half) +
    (BM * BN + BM * BN) * (int)sizeof(float) +
    (BM * kHeadDim) * (int)sizeof(float);

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// =============================================================================
// 1) preprocess：逐行算 LSE 与 delta=rowsum(dO∘O)
//   grid = (S, H, B)，block = THREADS
// =============================================================================
__global__ void preprocess_kernel(const __half* __restrict__ q,
                                  const __half* __restrict__ k,
                                  const __half* __restrict__ o,
                                  const __half* __restrict__ do_,
                                  float* __restrict__ delta,   // [B*S*H]
                                  float* __restrict__ lse,     // [B*S*H]
                                  int S, int H, float scale, int causal) {
  const int s = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const size_t row = ((size_t)(b * S + s)) * H + h;
  const __half* qr = q + row * kHeadDim;

  // --- online softmax：单趟求 (m, l)，无需物化整行 S ---
  float m = -INFINITY;
  float l = 0.f;
  const int jmax = causal ? (s + 1) : S;   // causal：只算 j <= s
  for (int j = tid; j < jmax; j += blockDim.x) {
    const __half* kr = k + (((size_t)(b * S + j)) * H + h) * kHeadDim;
    float dot = 0.f;
#pragma unroll 8
    for (int d = 0; d < kHeadDim; ++d)
      dot += __half2float(qr[d]) * __half2float(kr[d]);
    dot *= scale;
    float mn = fmaxf(m, dot);
    l = l * expf(m - mn) + expf(dot - mn);
    m = mn;
  }

  // --- block 归约 (m, l)：按 online-softmax 的合并规则 ---
  __shared__ float sh_m[THREADS];
  __shared__ float sh_l[THREADS];
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

  // --- delta = sum_d O∘dO ---
  float dp = 0.f;
  const __half* orow = o + row * kHeadDim;
  const __half* dorow = do_ + row * kHeadDim;
  for (int d = tid; d < kHeadDim; d += blockDim.x)
    dp += __half2float(orow[d]) * __half2float(dorow[d]);
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
// 2) main kernel：1colblock 反向
//   grid = (ceil(S/BM), H, B)，block = THREADS
//   dq_acc 直接写（每个 Q 块独占行）；dk_acc/dv_acc 用 atomicAdd。
// =============================================================================
__global__ void __launch_bounds__(THREADS)
fa_bwd_fp16_kernel(const __half* __restrict__ q,
                   const __half* __restrict__ k,
                   const __half* __restrict__ v,
                   const __half* __restrict__ do_,
                   const float* __restrict__ delta,
                   const float* __restrict__ lse,
                   float* __restrict__ dq_acc,
                   float* __restrict__ dk_acc,
                   float* __restrict__ dv_acc,
                   int S, int H, float scale, int causal) {
  extern __shared__ char smem[];
  __half* Qs  = reinterpret_cast<__half*>(smem);
  __half* Ks  = Qs + BM * kHeadDim;
  __half* Vs  = Ks + BN * kHeadDim;
  __half* dOs = Vs + BN * kHeadDim;
  float* Ss = reinterpret_cast<float*>(dOs + BM * kHeadDim);  // 先存 S，后覆盖为 dS
  float* Ps = Ss + BM * BN;
  float* dQs = Ps + BM * BN;                                   // dQ 的 smem 累加器

  const int mblk = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int m0 = mblk * BM;

  // ---- 载入本 Q 块的 Q 与 dO（越界补 0）----
  for (int i = tid; i < BM * kHeadDim; i += THREADS) {
    int r = i / kHeadDim, d = i % kHeadDim;
    int qi = m0 + r;
    __half qv = __float2half(0.f), ov = __float2half(0.f);
    if (qi < S) {
      size_t idx = (((size_t)(b * S + qi)) * H + h) * kHeadDim + d;
      qv = q[idx];
      ov = do_[idx];
    }
    Qs[i] = qv;
    dOs[i] = ov;
  }
  for (int i = tid; i < BM * kHeadDim; i += THREADS) dQs[i] = 0.f;
  __syncthreads();

  // causal：只需处理到本 Q 块最后一行的列；否则到 S。
  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;

    // ---- 载入 K/V 块 ----
    for (int i = tid; i < BN * kHeadDim; i += THREADS) {
      int j = i / kHeadDim, d = i % kHeadDim;
      int jg = j0 + j;
      __half kv = __float2half(0.f), vv = __float2half(0.f);
      if (jg < S) {
        size_t idx = (((size_t)(b * S + jg)) * H + h) * kHeadDim + d;
        kv = k[idx];
        vv = v[idx];
      }
      Ks[i] = kv;
      Vs[i] = vv;
    }
    __syncthreads();

    // ---- S = scale·QKᵀ，P = exp(S − LSE)；同一个 warp 负责 WM_ROWS 个 Q 行，
    //      lane 对应 BN 个列中的一列（BN=32=warpSize）----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      float dot = 0.f;
      if (qi < S) {
        const __half* qrow = Qs + r * kHeadDim;
        const __half* krow = Ks + lane * kHeadDim;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d)
          dot += __half2float(qrow[d]) * __half2float(krow[d]);
        dot *= scale;
      }
      int jg = j0 + lane;
      float p = 0.f;
      if (qi < S && jg < S && !(causal && jg > qi))
        p = expf(dot - lse[((size_t)(b * S + qi)) * H + h]);
      Ps[r * BN + lane] = p;
    }
    __syncthreads();

    // ---- dP = dO Vᵀ；dS = P∘(dP − D)（覆盖写入 Ss）----
#pragma unroll
    for (int rr = 0; rr < WM_ROWS; ++rr) {
      int r = warp * WM_ROWS + rr;
      int qi = m0 + r;
      float dp = 0.f;
      if (qi < S) {
        const __half* drow = dOs + r * kHeadDim;
        const __half* vrow = Vs + lane * kHeadDim;
#pragma unroll 8
        for (int d = 0; d < kHeadDim; ++d)
          dp += __half2float(drow[d]) * __half2float(vrow[d]);
      }
      float del = (qi < S) ? delta[((size_t)(b * S + qi)) * H + h] : 0.f;
      Ss[r * BN + lane] = Ps[r * BN + lane] * (dp - del);
    }
    __syncthreads();

    // ---- dV = Pᵀ dO：warp 负责 WN_ROWS 个 K 行，lane 覆盖 4 个 head_dim 槽 ----
#pragma unroll
    for (int jj = 0; jj < WN_ROWS; ++jj) {
      int j = warp * WN_ROWS + jj;
      int jg = j0 + j;
      if (jg >= S) continue;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
      for (int i = 0; i < BM; ++i) {
        float p = Ps[i * BN + j];          // 同一列，warp 内广播
        if (p == 0.f) continue;
        const __half* drow = dOs + i * kHeadDim;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += p * __half2float(drow[d]);
        }
      }
      float* base = dv_acc + (((size_t)(b * S + jg)) * H + h) * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) atomicAdd(base + lane + 32 * kk, acc[kk]);
    }

    // ---- dK = scale·dSᵀ Q（dS 在 Ss）----
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
        const __half* qrow = Qs + i * kHeadDim;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += ds * __half2float(qrow[d]);
        }
      }
      float* base = dk_acc + (((size_t)(b * S + jg)) * H + h) * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        atomicAdd(base + lane + 32 * kk, acc[kk] * scale);
    }

    // ---- dQ += scale·dS K：累积到 smem dQs（本 CTA 独占 Q 行，无需 atomic）----
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
        const __half* krow = Ks + j * kHeadDim;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          int d = lane + 32 * kk;
          acc[kk] += ds * __half2float(krow[d]);
        }
      }
      float* dqr = dQs + r * kHeadDim;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk)
        dqr[lane + 32 * kk] += acc[kk] * scale;
    }
    __syncthreads();
  }

  // ---- 写回 dQ（fp32 缓冲，稍后 convert）----
  for (int i = tid; i < BM * kHeadDim; i += THREADS) {
    int r = i / kHeadDim;
    int qi = m0 + r;
    if (qi < S)
      dq_acc[(((size_t)(b * S + qi)) * H + h) * kHeadDim + (i % kHeadDim)] = dQs[i];
  }
}

// =============================================================================
// 3) convert：fp32 累加缓冲 → fp16 输出
// =============================================================================
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc,
                               __half* __restrict__ dq,
                               __half* __restrict__ dk,
                               __half* __restrict__ dv,
                               size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    dq[i] = __float2half(dq_acc[i]);
    dk[i] = __float2half(dk_acc[i]);
    dv[i] = __float2half(dv_acc[i]);
  }
}

// =============================================================================
// 极简 npy 读取（只支持 little-endian C-contiguous float32）
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
    st.max_abs = std::max(st.max_abs, d);
    st.max_rel = std::max(st.max_rel, d / (std::fabs((double)b[i]) + 1e-3));
  }
  return st;
}

// =============================================================================
// host / launcher / self-test
// =============================================================================
int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16";
  std::string o_name = "ref_o";
  bool causal = true;
  int iters = 50;
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
  const float scale = 1.0f / sqrtf((float)D);

  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d D=%d causal=%d scale=%.6f\n", B, S, H, D, (int)causal, scale);

  // host 端 float -> half
  auto to_half = [&](const std::vector<float>& src) {
    std::vector<__half> h(src.size());
    for (size_t i = 0; i < src.size(); ++i) h[i] = __float2half(src[i]);
    return h;
  };
  auto qh = to_half(q_np.data), kh = to_half(k_np.data), vh = to_half(v_np.data),
       doh = to_half(do_np.data), oh = to_half(o_np.data);

  __half *dq, *dk, *dv;
  __half *d_q, *d_k, *d_v, *d_o;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc;
  CUDA_CHECK(cudaMalloc(&dq, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dk, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dv, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_k, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_v, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_o, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_delta, (size_t)B * S * H * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, (size_t)B * S * H * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, n * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, qh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, kh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, vh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o, oh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));

  dim3 pg(S, H, B);
  dim3 mg((S + BM - 1) / BM, H, B);
  const int cvt_threads = 256;
  const int cvt_blocks = (int)std::min<size_t>((n + cvt_threads - 1) / cvt_threads, 65535);

  // dO 单独一个指针（preprocess 需要 O 与 dO 两个输入）
  __half* d_do = nullptr;
  CUDA_CHECK(cudaMalloc(&d_do, n * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_do, doh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

  cudaEvent_t ev0, ev1, ev2;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventCreate(&ev2));

  auto warmup = [&]() {
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, n * sizeof(float)));
    preprocess_kernel<<<pg, THREADS>>>(d_q, d_k, d_o, d_do, d_delta, d_lse, S, H, scale,
                                       (int)causal);
    fa_bwd_fp16_kernel<<<mg, THREADS, SMEM_BYTES>>>(d_q, d_k, d_v, d_do, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                    scale, (int)causal);
    convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, dq, dk, dv, n);
  };
  for (int i = 0; i < 3; ++i) warmup();
  CUDA_CHECK(cudaDeviceSynchronize());

  // 计时（纯 device，三个 kernel 合计；CUDA event 口径）
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) warmup();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  ms /= iters;
  double flops = 4.0 * (double)B * S * H * S * D;
  double tflops = flops / (ms * 1e-3) / 1e12;
  printf("[timing] total(3 kernels) %.4f ms  %.2f TFLOPS (bwd FLOPs=4BS^2HD)\n", ms, tflops);

  // 单独测 preprocess / main
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i)
    preprocess_kernel<<<pg, THREADS>>>(d_q, d_k, d_o, d_do, d_delta, d_lse, S, H, scale,
                                       (int)causal);
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
    fa_bwd_fp16_kernel<<<mg, THREADS, SMEM_BYTES>>>(d_q, d_k, d_v, d_do, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                    scale, (int)causal);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_main = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_main, ev0, ev1));
  ms_main /= iters;
  printf("[timing] preprocess %.4f ms | main %.4f ms | convert %.4f ms\n", ms_pre, ms_main,
         ms - ms_pre - ms_main);

  // ---- 数值对拍：读回我们 kernel 的输出 ----
  std::vector<__half> hdq(n), hdk(n), hdv(n);
  CUDA_CHECK(cudaMemcpy(hdq.data(), dq, n * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdk.data(), dk, n * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdv.data(), dv, n * sizeof(__half), cudaMemcpyDeviceToHost));
  std::vector<float> mdq(n), mdk(n), mdv(n);
  for (size_t i = 0; i < n; ++i) {
    mdq[i] = __half2float(hdq[i]);
    mdk[i] = __half2float(hdk[i]);
    mdv[i] = __half2float(hdv[i]);
  }

  auto print_cmp = [&](const char* name, const std::vector<float>& mine,
                       const std::vector<float>& ref) {
    DiffStat st = diff_stat(mine, ref);
    printf("  %-4s vs ref: max_abs=%.3e  max_rel=%.3e\n", name, st.max_abs, st.max_rel);
  };
  printf("[compare] ours vs ref (O from %s.npy)\n", o_name.c_str());
  print_cmp("dq", mdq, rdq.data);
  print_cmp("dk", mdk, rdk.data);
  print_cmp("dv", mdv, rdv.data);

  // 也顺带比 ref 与 fa/te 的差距（如果有），方便判断容差是否合理
  auto try_cmp = [&](const char* fname, const std::vector<float>& ref) {
    std::ifstream test(dir + "/" + fname + ".npy");
    if (!test.good()) return;
    auto t = load_npy_f32(dir + "/" + fname + ".npy");
    DiffStat st = diff_stat(t.data, ref);
    printf("  %-9s vs ref: max_abs=%.3e  max_rel=%.3e\n", fname, st.max_abs, st.max_rel);
  };
  try_cmp("fa_dq", rdq.data);
  try_cmp("fa_dk", rdk.data);
  try_cmp("fa_dv", rdv.data);
  try_cmp("te_dq", rdq.data);
  try_cmp("te_dk", rdk.data);
  try_cmp("te_dv", rdv.data);

  cudaFree(dq); cudaFree(dk); cudaFree(dv);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o); cudaFree(d_do);
  cudaFree(d_delta); cudaFree(d_lse);
  cudaFree(d_dq_acc); cudaFree(d_dk_acc); cudaFree(d_dv_acc);
  return 0;
}
