// =============================================================================
// fa_bwd_bf16_mma_main.cu —— bf16 张量核反向（O5b）**两文件版的 host 部分**
// =============================================================================
// device 代码（mma/ldmatrix 封装、preprocess_kernel、fa_bwd_bf16_mma_kernel、
// convert_kernel）见 `fa_bwd_bf16_mma_kernels.cuh`；本文件只保留 host 侧：
// npy 读取 / launcher / 自测对拍。行为与单文件 `fa_bwd_bf16_mma_onefile.cu`
// **逐位一致**（device 代码逐字未改）。
//
// O6c：host 增加主 kernel tile/PIPE 自动档（小网格 BM=32、大 S BN=64）与 CLI
// `--bm/--bn/--pipe/--sched` 覆盖；与 fp16 版逐字同构。
//
// 用法：run.sh src/bf16/fa_bwd_bf16_mma_main.cu [--dir=...] [--full|--causal] [--iters=N]
// =============================================================================

#include "fa_bwd_bf16_mma_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#if defined(FA_WGMMA) && defined(FA_TMA)
// O31：为 LSE 的 Q/K 建 4D TMA 描述符（dims={D,S,H,B}，SW128，box={64,64,1,1}）。
// 与 fp16 O30 的 `make_lse_map` 逐字同构，仅 dtype 换成 `BFLOAT16`（同样 2B，box 内维
// 128B = 64 元素、stride 字节数不变）。坐标 {k0,row,head,batch}。
static CUtensorMap make_lse_map(const void* ptr, long long H, long long S, long long D,
                                long long B) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D * 2), (uint64_t)(D * 2),
                         (uint64_t)(S * H * D * 2)};
  uint32_t box[4] = {64, 64, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", s);
    std::exit(1);
  }
  return map;
}

// O34：主 kernel 用的大张量描述符（dims={D,S,H,B}，SW128）。box={64,8,1,1} —— 内维 128B
// （64 个 bf16）= SW128 跨距，8 行 = 一个 1024B atom；逐 atom TMA 即可原样写出 `sw128_off`
// 的交织布局（见 `tma_fill_sw128`）。与 fp16 O33 的 `make_main_map` 逐字节同构，仅 dtype
// 换成 `BFLOAT16`（同样 2B，字节 stride 不变）。
static CUtensorMap make_main_map(const void* ptr, long long H, long long S, long long D,
                                 long long B) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D * 2), (uint64_t)(D * 2),
                         (uint64_t)(S * H * D * 2)};
  uint32_t box[4] = {64, 8, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled(main) failed: %s\n", s);
    std::exit(1);
  }
  return map;
}
#endif

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
    st.max_abs = std::max(st.max_abs, d);
    st.max_rel = std::max(st.max_rel, d / (std::fabs((double)b[i]) + 1e-3));
  }
  return st;
}

// =============================================================================
// host / launcher / self-test
// =============================================================================
// PIPE=0：K/V 都不双缓冲；PIPE=1：K/V 都双缓冲；PIPE=2：只 K 双缓冲（V 单缓冲 + 后段预取）。
// K/V 的 smem 份数：0→2（各 1）、1→4（各 2）、2→3（K 2 + V 1）。
template <int HD, int BM, int BN, int PIPE, bool R4 = false, bool PREL = true>
static void launch_bwd_mma(dim3 mg, const bf16* q, const bf16* k, const bf16* v,
                           const bf16* do_, const float* delta, const float* lse,
                           float* dq_acc, float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                           float scale, int causal, int sched,
                           const int* cu_seqlens = nullptr) {
  constexpr int kvn = (PIPE == 0) ? 2 : (PIPE == 1 ? 4 : 3);
  constexpr int pds = (PIPE == 2) ? 2 * BM * (BN + 8) : 2 * BN * (BM + 8) + BM * (BN + 8);
  constexpr int smem =
      (2 * BM * (HD + 8) + kvn * BN * (HD + 8) + pds) * (int)sizeof(bf16);
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_bf16_mma_kernel<HD, BM, BN, PIPE, R4, PREL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_bf16_mma_kernel<HD, BM, BN, PIPE, R4, PREL><<<mg, THREADS, smem>>>(
      q, k, v, do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, sched,
      cu_seqlens);
}

#ifdef FA_WGMMA
// O9b：wgmma 主 kernel（只 HD=128 / BM=BN=64）。smem 见 kernel 内注释（≈99KB，2 CTA/SM）。
template <int HD>
static void launch_bwd_wgmma(dim3 mg, const bf16* q, const bf16* k, const bf16* v,
                             const bf16* do_, const float* delta, const float* lse,
                             float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                             int Hkv, float scale, int causal, int sched) {
  constexpr int BM = 64, BN = 64;
  constexpr int TILE  = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int smem = 1024 + TILE * 2 + KTILE * 3 + 2 * BM * BN * (int)sizeof(bf16);
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_bf16_wgmma_kernel<HD>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_bf16_wgmma_kernel<HD><<<mg, THREADS, smem>>>(q, k, v, do_, delta, lse, dq_acc,
                                                      dk_acc, dv_acc, S, H, Hkv, scale, causal,
                                                      sched);
}

// O17-bf16：2 warpgroup（BM=128）wgmma 主 kernel（只 HD=128）。Q/dO/K/V + P/dS 全 SW128。
// smem = 1024(对齐) + Q 32KB + dO 32KB + K 2×16KB + V 16KB + P 16KB + dS 16KB ≈ 145KB。
template <int HD, bool SPLIT = true>
static void launch_bwd_wgmma2(dim3 mg, const bf16* q, const bf16* k, const bf16* v,
                              const bf16* do_, const float* delta, const float* lse,
                              float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                              int Hkv, float scale, int causal, bf16* dq_h = nullptr,
                              const int* cu_seqlens = nullptr,
                              int ksplit = 1) {
  static_assert(HD == 128, "wgmma2 只做 HD=128");
  constexpr int BM = 128, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_bf16_wgmma2_kernel<HD, SPLIT>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_bf16_wgmma2_kernel<HD, SPLIT><<<mg, 256, smem>>>(q, k, v, do_, delta, lse, dq_acc,
                                                          dk_acc, dv_acc, S, H, Hkv, scale,
                                                          causal, dq_h, cu_seqlens, ksplit);
}

// O18-bf16：BN=128 版 wgmma2（只 HD=128）。tile 数减半；smem = 1024 + Q32 + dO32 + K 2×32 +
// V32 + P32 + dS32 ≈ 230400B（仍 1 CTA/SM）。
template <int HD, bool SPLIT = true>
static void launch_bwd_wgmma2b(dim3 mg, const bf16* q, const bf16* k, const bf16* v,
                               const bf16* do_, const float* delta, const float* lse,
                               float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                               int Hkv, float scale, int causal, bf16* dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2b 只做 HD=128");
  constexpr int BM = 128, BN = 128;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_bf16_wgmma2b_kernel<HD, SPLIT>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_bf16_wgmma2b_kernel<HD, SPLIT><<<mg, 256, smem>>>(q, k, v, do_, delta, lse, dq_acc,
                                                           dk_acc, dv_acc, S, H, Hkv, scale,
                                                           causal, dq_h);
}

#if defined(FA_TMA)
// O34：把 wgmma2b 的 Q/K/V/dO 载入换成逐 atom 4D-TMA（布局逐字节不变）。smem 与 wgmma2b 同
// （+ 4 个 mbarrier 32B），故 1 CTA/SM 不变。与 fp16 O33 的 launcher 逐字同构。
template <int HD, bool SPLIT = true>
static void launch_bwd_wgmma2b_tma(dim3 mg, CUtensorMap qmap, CUtensorMap kmap,
                                   CUtensorMap vmap, CUtensorMap dmap, const float* delta,
                                   const float* lse, float* dq_acc, float* dk_acc,
                                   float* dv_acc, int S, int H, int Hkv, float scale,
                                   int causal, bf16* dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2b TMA 只做 HD=128");
  constexpr int BM = 128, BN = 128;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2 + 128;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_bf16_wgmma2b_tma_kernel<HD, SPLIT>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_bf16_wgmma2b_tma_kernel<HD, SPLIT><<<mg, 256, smem>>>(
      qmap, kmap, vmap, dmap, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal,
      dq_h);
}

// O36：把 **BN=64** 的 wgmma2 主 kernel 的 Q/K/V/dO 换成逐 atom 4D-TMA（布局逐字节不变）。
// smem 与 cp.async 版 wgmma2 同（+ 4 个 mbarrier 32B），故 1 CTA/SM 不变。与 fp16 O35 同构。
template <int HD, bool SPLIT = true>
static void launch_bwd_wgmma2_tma(dim3 mg, CUtensorMap qmap, CUtensorMap kmap,
                                  CUtensorMap vmap, CUtensorMap dmap, const float* delta,
                                  const float* lse, float* dq_acc, float* dk_acc,
                                  float* dv_acc, int S, int H, int Hkv, float scale,
                                  int causal, bf16* dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2 TMA 只做 HD=128");
  constexpr int BM = 128, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2 + 128;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_bf16_wgmma2_tma_kernel<HD, SPLIT>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_bf16_wgmma2_tma_kernel<HD, SPLIT><<<mg, 256, smem>>>(
      qmap, kmap, vmap, dmap, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal,
      dq_h);
}
#endif  // FA_TMA
#endif  // FA_WGMMA

#ifdef FA_WGMMA
static int run_varlen(const std::string& dir, bool causal, int iters, int lse_split = 0,
                      int wg2ksplit = -1) {
  auto q_np = load_npy_f32(dir + "/q.npy");
  auto k_np = load_npy_f32(dir + "/k.npy");
  auto v_np = load_npy_f32(dir + "/v.npy");
  auto do_np = load_npy_f32(dir + "/do.npy");
  auto o_np = load_npy_f32(dir + "/ref_o.npy");
  auto rdq = load_npy_f32(dir + "/ref_dq.npy");
  auto rdk = load_npy_f32(dir + "/ref_dk.npy");
  auto rdv = load_npy_f32(dir + "/ref_dv.npy");
  auto cu_np = load_npy_f32(dir + "/cu_seqlens.npy");
  if (q_np.shape.size() != 3 || k_np.shape.size() != 3) {
    fprintf(stderr, "VARLEN 期望 q 为 [T,H,D]、k 为 [T,Hkv,D]\n");
    return 1;
  }
  const int T = (int)q_np.shape[0], H = (int)q_np.shape[1], D = (int)q_np.shape[2];
  const int Hkv = (int)k_np.shape[1];
  const int B = (int)cu_np.data.size() - 1;
  if (D != 128 && D != 512) {
    fprintf(stderr, "VARLEN 目前只做 HD=128/512；当前 %d\n", D);
    return 1;
  }
  int maxlen = 0;
  for (int b = 0; b < B; ++b) {
    int L = (int)cu_np.data[b + 1] - (int)cu_np.data[b];
    if (L > maxlen) maxlen = L;
  }
  const size_t nq = (size_t)T * H * D;
  const size_t nkv = (size_t)T * Hkv * D;
  const size_t rows_q = (size_t)T * H;
  const float scale = 1.0f / sqrtf((float)D);
  printf("case = %s\n", dir.c_str());
  printf("VARLEN: B=%d T=%d maxlen=%d H=%d Hkv=%d D=%d causal=%d\n", B, T, maxlen, H, Hkv, D,
         (int)causal);
  printf("cu_seqlens =");
  for (int b = 0; b <= B && b < 12; ++b) printf(" %d", (int)cu_np.data[b]);
  printf("%s\n", (B > 11) ? " ..." : "");

  auto to_half = [&](const std::vector<float>& src) {
    std::vector<bf16> h(src.size());
    for (size_t i = 0; i < src.size(); ++i) h[i] = __float2bfloat16(src[i]);
    return h;
  };
  auto qh = to_half(q_np.data), kh = to_half(k_np.data), vh = to_half(v_np.data),
       doh = to_half(do_np.data), oh = to_half(o_np.data);

  bf16 *dq, *dk, *dv, *d_q, *d_k, *d_v, *d_o, *d_do;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc;
  int* d_cu;
  CUDA_CHECK(cudaMalloc(&dq, nq * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dk, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dv, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_q, nq * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_k, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_v, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_o, nq * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_do, nq * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_delta, rows_q * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, rows_q * sizeof(float)));
  // O40：varlen LSE 的 K 维 split 部分结果（[T*H][ksplit] 个 (m,l)），按最大 split=16 预留。
  float* d_lse_part = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lse_part, rows_q * 16 * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, nq * sizeof(float)));   // dq_h 路径下不用，占位
  CUDA_CHECK(cudaMalloc(&d_dk_acc, nkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, nkv * sizeof(float)));
  std::vector<int> cu(B + 1);
  for (int b = 0; b <= B; ++b) cu[b] = (int)cu_np.data[b];
  CUDA_CHECK(cudaMalloc(&d_cu, (B + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_cu, cu.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_q, qh.data(), nq * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, kh.data(), nkv * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, vh.data(), nkv * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do, doh.data(), nq * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o, oh.data(), nq * sizeof(bf16), cudaMemcpyHostToDevice));

  const int lse_nblk = (maxlen + LBM - 1) / LBM;
  dim3 lg_bal((lse_nblk + 1) / 2, H, B);
  // non-causal 用 O8 的 mma `lse_mma_kernel<HD>`；causal D=512 用 `lse_mma_kernel_bal<512,1>`
  // （镜像配对 + cu_seqlens，PIPE=1 双缓冲）。
  const int kLseSmem = (LBM + LBN) * (D + 8) * (int)sizeof(bf16);
  const int kLseSmemBal1 = (LBM + 2 * LBN) * (D + 8) * (int)sizeof(bf16);
  const int kLseTileWgm = (D == 128) ? (LBM / 8) * (D / 64) * 1024 : 0;
  const int kLseSmemWgm1 = 1024 + kLseTileWgm * 3;
  if (D == 128) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_wgmma<128, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kLseSmemWgm1));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<128>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
  } else {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<512>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<512, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal1));
  }
  const int d_rows = (int)rows_q;
  const int d_wpb = THREADS / 32;
  const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
  const int main_bm = (D == 512) ? 32 : 128;   // D=512：MLA 主 kernel 几何（BM=32）
  dim3 mg((maxlen + main_bm - 1) / main_bm, H, B);

  // O43：varlen 主 kernel（D==128 的 wgmma2）N 方向 split-K。本轮实测短序列 varlen 切 K 反慢
  //   （Q/dO 重载 + 额外 convert 抵不过填 SM），故**不做 auto**，仅 `--wg2ksplit=N>=2` 显式开启。
  int wg2_ks = 1;
  if (D == 128 && wg2ksplit >= 2) { wg2_ks = wg2ksplit; mg.x *= (unsigned)wg2_ks; }
  const bool vq_direct = (D == 128) && (wg2_ks <= 1);

  // O40：varlen LSE 的 K 维 split auto（D=128 目标 `grid*split≈528`=一个波、cap 8；D=512
  //   `≈132`、cap 16，与 O38/O39 同标定）；`--lsesplit=N`（>0）直接指定。再按最大序列 tile 数封顶。
  const int lse_nblk0 = (maxlen + LBM - 1) / LBM;
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    long base = (long)((lse_nblk0 + 1) / 2) * H * B;
    const int target = (D == 512) ? 132 : 528;
    const int cap = (D == 512) ? 16 : 8;
    int sp = 1;
    while (sp < cap && base * (sp * 2) <= target) sp *= 2;
    while (sp > lse_nblk0 && sp > 1) sp >>= 1;
    lse_split_eff = sp;
  }
  printf("[O40] varlen lse k-split = %d (base=%ld)\n", lse_split_eff,
         (long)((lse_nblk0 + 1) / 2) * H * B);

  const int cvt_threads = 256;
  const int cvt_blocks =
      (int)std::min<size_t>((std::max(nq, nkv) + cvt_threads - 1) / cvt_threads, 65535);

  auto run_all = [&]() {
    if (D == 128 && wg2_ks > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
    if (D == 128) {
      if (causal) {
        if (lse_split_eff > 1) {
          // O40：K 维切片 + 二次归约。
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal_wgmma<128, 1><<<gsp, THREADS, kLseSmemWgm1>>>(
              d_q, d_k, d_lse, maxlen, H, Hkv, scale, d_cu, d_lse_part, lse_split_eff);
          const long long nrows = (long long)rows_q;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(
              d_q, d_k, d_lse, maxlen, H, Hkv, scale, d_cu);
        }
      } else
        lse_mma_kernel<128><<<dim3(lse_nblk, H, B), THREADS, kLseSmem>>>(d_q, d_k, d_lse,
                                                                         maxlen, H, Hkv, scale,
                                                                         0, d_cu);
      delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      // O24：dQ 由主 kernel 直接写 bf16（dq_h），convert 跳过 dQ（n_q=0）。
      // O43：ksplit>1 时 dQ 跨 CTA 原子累加 ⇒ dq_h 传 nullptr、convert 转全部 nq/nkv。
      bf16* dqo = vq_direct ? dq : nullptr;
      launch_bwd_wgmma2<128, true>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc,
                                   d_dv_acc, maxlen, H, Hkv, scale, (int)causal, dqo, d_cu,
                                   wg2_ks);
      convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, dq, dk, dv,
                                                  vq_direct ? 0 : nq, nkv);
    } else {
      // HD=512（MLA）：causal LSE 走 mma 镜像配对版（带 cu_seqlens）；dQ 全局累加（NDT>1）
      // ⇒ 先清零 dq_acc，convert 转全部 nq/nkv。
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * sizeof(float)));
      if (causal) {
        if (lse_split_eff > 1) {
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal<512, 1><<<gsp, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, maxlen, H, Hkv, scale, d_cu, d_lse_part, lse_split_eff);
          const long long nrows = (long long)rows_q;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal<512, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, maxlen, H, Hkv, scale, d_cu);
        }
      } else
        lse_mma_kernel<512><<<dim3(lse_nblk, H, B), THREADS, kLseSmem>>>(d_q, d_k, d_lse,
                                                                         maxlen, H, Hkv, scale,
                                                                         0, d_cu);
      delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      launch_bwd_mma<512, 32, 32, 1, false, true>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse,
                                                  d_dq_acc, d_dk_acc, d_dv_acc, maxlen, H, Hkv,
                                                  scale, (int)causal, 0, d_cu);
      convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, dq, dk, dv, nq,
                                                  nkv);
    }
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
  double flops = 0.0;
  for (int b = 0; b < B; ++b) {
    double L = cu[b + 1] - cu[b];
    flops += 4.0 * H * L * L * D;  // 反向 ≈ 2×fwd，因果再乘系数（口径同 fp8 varlen）
  }
  printf("[timing] VARLEN total %.4f ms  %.2f TFLOPS (sum_b 4HL^2D)\n", ms,
         flops / (ms * 1e-3) / 1e12);
  printf("grid main = %d x %d x %d | lse grid = %d x %d x %d | T=%d\n",
         (maxlen + main_bm - 1) / main_bm, H, B, (lse_nblk + 1) / 2, H, B, T);

  std::vector<float> h_dq(nq), h_dk(nkv), h_dv(nkv);
  std::vector<bf16> h_dq_h(nq), h_dk_h(nkv), h_dv_h(nkv);
  CUDA_CHECK(cudaMemcpy(h_dq_h.data(), dq, nq * sizeof(bf16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dk_h.data(), dk, nkv * sizeof(bf16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dv_h.data(), dv, nkv * sizeof(bf16), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nq; ++i) h_dq[i] = __bfloat162float(h_dq_h[i]);
  for (size_t i = 0; i < nkv; ++i) { h_dk[i] = __bfloat162float(h_dk_h[i]); h_dv[i] = __bfloat162float(h_dv_h[i]); }
  auto report = [](const char* nm, const std::vector<float>& a, const NpyF32& b) {
    size_t n = std::min(a.size(), b.data.size());
    double ma = 0.0, mr = 0.0, ao = 0.0, ar = 0.0;
    size_t arg = 0;
    for (size_t i = 0; i < n; ++i) {
      double d = fabs((double)a[i] - (double)b.data[i]);
      if (d > ma) { ma = d; arg = i; }
      ao = std::max(ao, fabs((double)a[i]));
      ar = std::max(ar, fabs((double)b.data[i]));
      double den = std::max(1e-3, fabs((double)b.data[i]));
      double r = d / den;
      if (r > mr) mr = r;
    }
    printf("  %-3s vs ref: max_abs=%.3e  max_rel=%.3e  (ours_amax=%.3e ref_amax=%.3e @%zu)\n",
           nm, ma, mr, ao, ar, arg);
  };
  printf("[compare] VARLEN ours vs fp32 ref\n");
  report("dq", h_dq, rdq);
  report("dk", h_dk, rdk);
  report("dv", h_dv, rdv);
  return 0;
}
#endif  // FA_WGMMA

int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_bf16";
  std::string o_name = "ref_o";
  bool causal = true;
  // pipe: -1=自动（按网格大小选 1/2）、0=O5、1=O6、2=O6b。
  int pipe = -1;
  // O6c：causal 下 mblk 重排（0=原样，1=交错，2=逆序）。
  int sched = 0;
  // O6c：主 kernel tile 配置覆盖（-1=自动）。
  int bm_opt = -1, bn_opt = -1;
  // O7c：dK/dV 归约宽度（-1=自动/float2，0=float2，1=float4）。
  int r4_opt = -1;
  // O7c：LSE/D 预装寄存器（-1=自动/开，0=关，1=开）。
  int prel_opt = -1;
  // O9：LSE 是否用 wgmma（仅 D==128 且 causal；0=用 O8b 的 mma 版，1=wgmma 版）。
  int lse_wgm = 0;
  // O23：lse_wgm 是否被用户显式指定（--lsewgm=0/1）。未指定时在 FA_WGMMA 构建下默认开。
  bool lse_forced = false;
  // O31：LSE 是否用 TMA 版（仅 FA_TMA 构建、D==128、causal；1=用 4D TMA 载入 Q/K）。
  //   -1=自动（FA_TMA 构建下 D==128/causal 默认开），0/1 由 `--lsetma=` 强制。
  int lse_tma = -1;
  // O38（bf16 版）：LSE 的 K 维 split（0=auto，1=关；>1=切片数）。仅 TMA/causal 路径生效。
  int lse_split = 0;
  // O9b：主 kernel 是否用 wgmma（仅 FA_WGMMA 构建、D==128 且 sel=(64,64) 时生效）。
  int wgmma_sel = 0;
  // O17：2 warpgroup（BM=128，跨 wg 归约）wgmma 主 kernel（仅 FA_WGMMA 构建、D==128）。
  // O23：默认自动选择（见下方 auto 段）；`--wg2=0/1` 可强制。wg_forced=用户显式指定。
  int wg2_sel = 0;
  // O18：BN=128 版 wgmma2（仅 FA_WGMMA 构建、D==128）。O23：默认自动选择。
  int wg2bn_sel = 0;
  bool wg_forced = false;
  // O34：主 kernel（wgmma2b 几何）是否用逐 atom 4D-TMA 载入 Q/K/V/dO（仅 FA_TMA 构建、
  //   D==128、wg2bn 路径）。opt-in（`--maintma=0/1`），与 cp.async 版同 binary A/B。
  int maintma_sel = 0;
  // O43：wgmma2（BN=64）主 kernel 的 N 方向 split-K。`-1`=自动（仅 D==128、非 maintma、未切块
  //   grid 不足一个波时按需切）；`1`=关（A/B）；`>=2`=强制。
  int wg2ksplit = -1;
  // O17-2：wgmma2 的 GEMM3/GEMM4 是否拆分到两个 warpgroup（1=拆，0=原版 wg0 串行）。
  int wg2split_sel = 1;
  // O24：delta 用 warp-per-row 向量化版（1，默认）还是旧 block-per-row smem 版（0，A/B）。
  int delta_warp_sel = 1;
  // O24：D==128 wgmma2/2b 路径直接用 bf16 写 dQ、convert 跳过 dQ（1，默认；0=A/B）。
  int dq_direct_sel = 1;
  int varlen = 0;   // VARLEN：packed [T,H,D] + cu_seqlens.npy（bf16/HD=128/causal|full/wgmma2）
  int iters = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--varlen") varlen = 1;
    else if (a.rfind("--varlen=", 0) == 0) varlen = atoi(a.c_str() + 9);
    else if (a == "--full") causal = false;
    else if (a == "--causal") causal = true;
    else if (a == "--nopipe") pipe = 0;
    else if (a == "--pipe") pipe = 1;
    else if (a == "--pipe2") pipe = 2;
    else if (a.rfind("--sched=", 0) == 0) sched = atoi(a.c_str() + 8);
    else if (a.rfind("--bm=", 0) == 0) bm_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--bn=", 0) == 0) bn_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--r4=", 0) == 0) r4_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--prel=", 0) == 0) prel_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--lsewgm=", 0) == 0) { lse_wgm = atoi(a.c_str() + 9); lse_forced = true; }
    else if (a == "--lsewgm") { lse_wgm = 1; lse_forced = true; }
    else if (a.rfind("--lsetma=", 0) == 0) lse_tma = atoi(a.c_str() + 9);
    else if (a == "--lsetma") lse_tma = 1;
    else if (a.rfind("--lsesplit=", 0) == 0) lse_split = atoi(a.c_str() + 11);
    else if (a.rfind("--wgmma=", 0) == 0) wgmma_sel = atoi(a.c_str() + 8);
    else if (a == "--wgmma") wgmma_sel = 1;
    else if (a.rfind("--wg2=", 0) == 0) { wg2_sel = atoi(a.c_str() + 6); wg_forced = true; }
    else if (a.rfind("--wg2split=", 0) == 0) wg2split_sel = atoi(a.c_str() + 11);
    else if (a == "--wg2") { wg2_sel = 1; wg_forced = true; }
    else if (a.rfind("--wg2bn=", 0) == 0) { wg2bn_sel = atoi(a.c_str() + 8); wg_forced = true; }
    else if (a == "--wg2bn") { wg2bn_sel = 1; wg_forced = true; }
    else if (a.rfind("--maintma=", 0) == 0) maintma_sel = atoi(a.c_str() + 10);
    else if (a == "--maintma") maintma_sel = 1;
    else if (a.rfind("--wg2ksplit=", 0) == 0) wg2ksplit = atoi(a.c_str() + 12);
    else if (a.rfind("--deltawarp=", 0) == 0) delta_warp_sel = atoi(a.c_str() + 12);
    else if (a.rfind("--dqdirect=", 0) == 0) dq_direct_sel = atoi(a.c_str() + 11);
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  if (varlen) {
#ifdef FA_WGMMA
    return run_varlen(dir, causal, iters, lse_split, wg2ksplit);
#else
    fprintf(stderr, "VARLEN 需要 -DFA_WGMMA（sm_90a）构建\n");
    return 1;
#endif
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
  const int Hkv = (int)k_np.shape[2];
  if (D != 128 && D != 512) {
    fprintf(stderr, "O5b bf16 mma 版支持 head_dim=128/512；当前 %d\n", D);
    return 1;
  }
  if (H % Hkv != 0) {
    fprintf(stderr, "H(%d) 必须是 Hkv(%d) 的整数倍\n", H, Hkv);
    return 1;
  }
  const size_t n = (size_t)B * S * H * D;
  const size_t nkv = (size_t)B * S * Hkv * D;
  const float scale = 1.0f / sqrtf((float)D);

  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d Hkv=%d D=%d causal=%d scale=%.6f\n", B, S, H, Hkv, D, (int)causal,
         scale);

  auto to_half = [&](const std::vector<float>& src) {
    std::vector<bf16> h(src.size());
    for (size_t i = 0; i < src.size(); ++i) h[i] = __float2bfloat16(src[i]);
    return h;
  };
  auto qh = to_half(q_np.data), kh = to_half(k_np.data), vh = to_half(v_np.data),
       doh = to_half(do_np.data), oh = to_half(o_np.data);

  bf16 *dq, *dk, *dv;
  bf16 *d_q, *d_k, *d_v, *d_o, *d_do;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc;
  float* d_lse_part = nullptr;
  CUDA_CHECK(cudaMalloc(&dq, n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dk, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&dv, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_k, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_v, nkv * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_o, n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_do, n * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&d_delta, (size_t)B * S * H * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, (size_t)B * S * H * sizeof(float)));
  // O38：LSE K-split 的部分 (m,l)，每行最多 16 个 split × 2 个 fp32。
  CUDA_CHECK(cudaMalloc(&d_lse_part, (size_t)B * S * H * 16 * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, nkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, nkv * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, qh.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, kh.data(), nkv * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, vh.data(), nkv * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o, oh.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do, doh.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));

  dim3 pg(S, H, B);
  dim3 mg((S + 63) / 64, H, B);
  const int lse_nblk = (S + LBM - 1) / LBM;
  dim3 lg(lse_nblk, H, B);
  dim3 lg_bal((lse_nblk + 1) / 2, H, B);   // O8b：镜像配对，grid.x 减半
  // LSE smem 随 head_dim 变化：Qs[LBM*LD] +（PIPE=0 时 1 份 / PIPE=1 时 2 份）Ks[LBN*LD]，LD=HD+8。
  const int LDl = D + 8;
  const int kLseSmem = (LBM + LBN) * LDl * (int)sizeof(bf16);
  // O8b：PIPE=0 单缓冲（与 O8 同尺寸），PIPE=1 双缓冲。
  const int kLseSmemBal0 = (LBM + LBN) * LDl * (int)sizeof(bf16);
  const int kLseSmemBal1 = (LBM + 2 * LBN) * LDl * (int)sizeof(bf16);
  // O9：wgmma LSE（仅 D=128）用 SW128 tile：TILE=(LBM/8)*(D/64)*1024 B，另加 1024B 对齐余量。
  const int kLseTileWgm = (LBM / 8) * (D / 64) * 1024;
  const int kLseSmemWgm0 = 1024 + kLseTileWgm * 2;  // Q + K（单缓冲）
  const int kLseSmemWgm1 = 1024 + kLseTileWgm * 3;  // Q + 2×K
  const int kLseSmemTma1 = 1024 + kLseTileWgm * 3 + 64;  // Q + 2×K + 2 个 mbarrier
  if (D == 128) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_wgmma<128, 0>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kLseSmemWgm0));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_wgmma<128, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kLseSmemWgm1));
#if defined(FA_WGMMA) && defined(FA_TMA)
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<128, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kLseSmemTma1));
#endif
  }
#if defined(FA_WGMMA) && defined(FA_TMA)
  // O31：建 LSE 的 Q/K 4D TMA 描述符（一次，供所有 (h,b) CTA 用坐标选择）。
  CUtensorMap qmap_lse, kmap_lse;
  if (D == 128) {
    qmap_lse = make_lse_map(d_q, H, S, D, B);
    kmap_lse = make_lse_map(d_k, Hkv, S, D, B);
  }
  // O34：主 kernel 的 Q/K/V/dO 描述符（box={64,8}，逐 atom）。
  CUtensorMap qmap_m, kmap_m, vmap_m, dmap_m;
  if (D == 128) {
    qmap_m = make_main_map(d_q, H, S, D, B);
    kmap_m = make_main_map(d_k, Hkv, S, D, B);
    vmap_m = make_main_map(d_v, Hkv, S, D, B);
    dmap_m = make_main_map(d_do, H, S, D, B);
  }
#endif
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<128>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<128, 0>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal0));
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<128, 1>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal1));
  if (D == 512) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<512>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<512, 0>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal0));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<512, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal1));
  }
  const int cvt_threads = 256;
  const int cvt_blocks =
      (int)std::min<size_t>((std::max(n, nkv) + cvt_threads - 1) / cvt_threads, 65535);

  // O6c：主 kernel 的 tile/PIPE 自动选择。
  //  - grid < 132（不到「每 SM 一个 CTA」）且 S 较小时，把 BM 减半到 32 → grid 翻倍、
  //    并行度翻倍（S=512 MHA：main 0.0876→0.0792ms，1.11×）。大 S 下 BM=32 会让
  //    dK/dV 的跨 CTA 原子量翻倍，故只在 S≤1024 用。
  //  - 否则 BM=64：大网格走 O6b（PIPE=2，只双缓冲 K）；小网格走 O6（PIPE=1）。
  // O13：auto tile 重新标定（O7c-PREL 之后，见 fp16 版注释）。
  const long long grid = (long long)((S + 63) / 64) * H * B;
  const bool bn64 = (S >= 4096) || (grid <= 256) || (grid > 600);
  int auto_bm = 64;
  int auto_bn = bn64 ? 64 : 32;
  int auto_pipe = (grid >= 396) ? 2 : 1;
  if (D == 512) {
    // MLA（HD=512）：BM=64 时 K/V 双缓冲会超 smem；BM=32 + PIPE=1 仍 ≤232KB 且最快。
    auto_bm = 32;
    auto_bn = 32;
    auto_pipe = 1;
  }
  const int bm_sel = (bm_opt > 0) ? bm_opt : auto_bm;
  const int bn_sel = (bn_opt > 0) ? bn_opt : auto_bn;
  const int pp_sel = (pipe >= 0) ? pipe : auto_pipe;
  printf("[O6c] main grid=%lld auto=(BM=%d,BN=%d,PIPE=%d) sel=(BM=%d,BN=%d,PIPE=%d)\n", grid,
         auto_bm, auto_bn, auto_pipe, bm_sel, bn_sel, pp_sel);
#define LAUNCH_CFG(HD_, BM_, BN_, PIPE_)                                                   \
  do {                                                                                     \
    if (r4) {                                                                              \
      if (prel)                                                                            \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, true, true>(g, d_q, d_k, d_v, d_do, d_delta,  \
                                                         d_lse, d_dq_acc, d_dk_acc,        \
                                                         d_dv_acc, S, H, Hkv, scale,       \
                                                         (int)causal, sched);              \
      else                                                                                 \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, true, false>(g, d_q, d_k, d_v, d_do, d_delta, \
                                                          d_lse, d_dq_acc, d_dk_acc,       \
                                                          d_dv_acc, S, H, Hkv, scale,      \
                                                          (int)causal, sched);             \
    } else {                                                                               \
      if (prel)                                                                            \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, false, true>(g, d_q, d_k, d_v, d_do, d_delta, \
                                                          d_lse, d_dq_acc, d_dk_acc,       \
                                                          d_dv_acc, S, H, Hkv, scale,      \
                                                          (int)causal, sched);             \
      else                                                                                 \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, false, false>(                            \
            g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,    \
            Hkv, scale, (int)causal, sched);                                              \
    }                                                                                      \
  } while (0)

  auto launch_cfg = [&](int bm, int bn, int pp, bool r4, bool prel) {
    dim3 g((S + bm - 1) / bm, H, B);
    if (D == 512) {
      // MLA：BN=32；BM 可 32/64；只支持 PIPE=0/1。
      if (bm == 64) {
        if (pp == 1) LAUNCH_CFG(512, 64, 32, 1);
        else LAUNCH_CFG(512, 64, 32, 0);
      } else {
        if (pp == 1) LAUNCH_CFG(512, 32, 32, 1);
        else LAUNCH_CFG(512, 32, 32, 0);
      }
      return;
    }
    if (bm == 32) {
      if (pp == 2) LAUNCH_CFG(128, 32, 32, 2);
      else if (pp == 1) LAUNCH_CFG(128, 32, 32, 1);
      else LAUNCH_CFG(128, 32, 32, 0);
    } else if (bn == 64) {
      LAUNCH_CFG(128, 64, 64, 2);
    } else if (pp == 2) {
      LAUNCH_CFG(128, 64, 32, 2);
    } else if (pp == 1) {
      LAUNCH_CFG(128, 64, 32, 1);
    } else {
      LAUNCH_CFG(128, 64, 32, 0);
    }
  };
#undef LAUNCH_CFG
  const bool r4_sel = (r4_opt > 0);
  const bool prel_sel = (prel_opt >= 0) ? (prel_opt != 0) : true;
  // O23：把 O17/O18 的 Hopper wgmma2 主 kernel 在 `-DFA_WGMMA`（sm_90a）构建下**默认打开**
  //   （对齐 fp8 的 O22）。仅 D==128：S>=4096 用 BN=128（O18，tile 数减半），否则 BN=64（O17）。
  //   用户显式传 `--wg2=`/`--wg2bn=` 时不做自动选择；`--wg2=0 --wg2bn=0` 退回 mma 路径做 A/B。
  //   非 FA_WGMMA 构建行为完全不变（wg2_sel/wg2bn_sel 恒 0）。
  // O43：wgmma2(BN=64) 主 kernel 的 N 方向 split-K 生效值（仅定长 D==128 的 cp.async 路径）。
  int wg2_ksplit_eff = 1;
#ifdef FA_WGMMA
  if (!wg_forced && D == 128) {
    if (S >= 4096) wg2bn_sel = 1; else wg2_sel = 1;
  }
  // O23：LSE 预处理也默认走 Hopper wgmma 版（仅 causal / D==128；非 causal 自动落回 O8 原版）。
  if (!lse_forced && D == 128 && causal) lse_wgm = 1;
  // O43：只在「BN=64 的 wgmma2、非 TMA」上切 K。小 S（base grid < 132）按需「填满一个波」；
  //   `--wg2ksplit=N` 可强制/关闭（=1）。大 S / BN=128 网格已够，不切。
  if (D == 128 && wg2_sel && !wg2bn_sel && !maintma_sel) {
    const int base = (int)((S + 127) / 128) * H * B;
    if (wg2ksplit >= 1) wg2_ksplit_eff = wg2ksplit;
    else {
      int sp = 1;
      while (sp < 8 && base * (sp * 2) <= 132) sp *= 2;
      int nt_cap = (S + 63) / 64;
      while (sp > nt_cap && sp > 1) sp >>= 1;
      wg2_ksplit_eff = sp;
    }
  }
#endif
  // O31：TMA 版 LSE 需驱动 API（`cuTensorMapEncodeTiled`）⇒ 只有 `-DFA_TMA -lcuda` 构建才编译
  //   该路径；此时 D==128/causal 默认开（对齐 fp16 O30 的做法，wgmma+cp.async → 4D TMA）。
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (lse_tma < 0) lse_tma = (D == 128 && causal) ? 1 : 0;
#else
  if (lse_tma < 0) lse_tma = 0;
#endif
  // O38（bf16 版）：自动 split 档（0=auto）。目标 `grid*split ≈ 528`（= 4 CTA/SM × 132 SM，
  //   即「填满一个波」；实测 bf16 LSE TMA 的 smem ~50KB ⇒ 4 CTA/SM），上限 8；grid 已达一个
  //   波则退回 1（逐位）。仅 TMA 路径生效。与 fp8 的 `≈2048` 不同：bf16 的 LSE 并行度更高、
  //   且大 S 时 512 CTA 已铺满一个波（S4096 split>1 反而慢 3–7%），故以「一波」为准。
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    long lg_grid = (long)lg_bal.x * H * B;
    // O39：D=512（MLA，mma LSE）目标 `grid*split ≈ 256`、上限 16；D=128 的 TMA LSE 维持
    //   O38 的「一波」目标 528、上限 8。
    const int target = (D == 512) ? 132 : 528;
    const int cap = (D == 512) ? 16 : 8;
    int sp = 1;
    while (sp < cap && lg_grid * (sp * 2) <= target) sp *= 2;
    // 再按 `nblk=ceil(S/64)` 封顶：切得比 K tile 数还细只会产生空切片 + 每 CTA 的 Q 载入开销。
    int nblk_cap = (S + 63) / 64;
    while (sp > nblk_cap) sp >>= 1;
    lse_split_eff = sp;
  }
  printf("[O38] lse k-split = %d%s\n", lse_split_eff, (lse_split <= 0 ? " (auto)" : ""));
  printf("[O23] main backend = %s | lse = %s (D=%d S=%d)%s\n",
         wg2bn_sel ? "wgmma2b(BN=128)" : (wg2_sel ? "wgmma2(BN=64)" : "mma"),
         (D == 128 && causal && (lse_tma ? 1 : lse_wgm)) ? (lse_tma ? "tma" : "wgmma") : "mma",
         D, S, (wg2_ksplit_eff > 1) ? " +ksplit" : "");
  printf("[O43] wgmma2 k-split = %d\n", wg2_ksplit_eff);
  bool dq_direct = false;
  auto run_main = [&]() {
#ifdef FA_WGMMA
    if (wg2bn_sel && D == 128) {
      dim3 g((S + 127) / 128, H, B);
      dq_direct = dq_direct_sel;
      bf16* dqo = dq_direct_sel ? dq : nullptr;
#if defined(FA_TMA)
      if (maintma_sel) {
        if (wg2split_sel)
          launch_bwd_wgmma2b_tma<128, true>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                            d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                            (int)causal, dqo);
        else
          launch_bwd_wgmma2b_tma<128, false>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                             d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                             (int)causal, dqo);
        return;
      }
#endif
      if (wg2split_sel)
        launch_bwd_wgmma2b<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo);
      else
        launch_bwd_wgmma2b<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo);
      return;
    }
    if (wg2_sel && D == 128) {
      dim3 g((S + 127) / 128, H, B);
      // O43：ksplit>1 时 dQ 跨 CTA 原子累加、不能直写 bf16（TMA 分支的 wg2_ksplit_eff 恒 1）。
      dq_direct = dq_direct_sel && (wg2_ksplit_eff <= 1);
      bf16* dqo = dq_direct ? dq : nullptr;
#if defined(FA_TMA)
      // O36：BN=64 的 wgmma2 也走逐 atom 4D-TMA（`--maintma`）。
      if (maintma_sel) {
        if (wg2split_sel)
          launch_bwd_wgmma2_tma<128, true>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                           d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                           (int)causal, dqo);
        else
          launch_bwd_wgmma2_tma<128, false>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                            d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                            (int)causal, dqo);
        return;
      }
#endif
      g.x *= (unsigned)wg2_ksplit_eff;   // O43：N 方向 split-K 抬高 grid
      if (wg2split_sel)
        launch_bwd_wgmma2<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo,
                                     nullptr, wg2_ksplit_eff);
      else
        launch_bwd_wgmma2<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo,
                                      nullptr, wg2_ksplit_eff);
      return;
    }
    if (wgmma_sel && D == 128 && bm_sel == 64 && bn_sel == 64) {
      dim3 g((S + 63) / 64, H, B);
      launch_bwd_wgmma<128>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc,
                            d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
      return;
    }
#endif
    launch_cfg(bm_sel, bn_sel, pp_sel, r4_sel, prel_sel);
  };
  // O24：delta 的 warp-per-row 版；`--deltawarp=0` 回旧版 A/B。行数 = B*S*H。
  const int d_rows = S * H * B;
  const int d_wpb = THREADS / 32;
  const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
  auto run_pre = [&]() {
    if (D == 512) {
      // O39：MLA（D=512）causal LSE 也用 K 维 split（此前只有 D=128/TMA 有 O38）。
      if (causal) {
        if (lse_split_eff > 1) {
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal<512, 1><<<gsp, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale, nullptr, d_lse_part, lse_split_eff);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal<512, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                        Hkv, scale);
        }
      } else
        lse_mma_kernel<512><<<lg, THREADS, kLseSmem>>>(d_q, d_k, d_lse, S, H, Hkv, scale,
                                                       (int)causal);
      if (delta_warp_sel)
        delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      else
        delta_kernel<512><<<pg, THREADS>>>(d_o, d_do, d_delta, S, H);
    } else {
#if defined(FA_WGMMA) && defined(FA_TMA)
      if (causal && lse_tma) {
        if (lse_split_eff > 1) {
          // O38：K 维切片 + 二次归约。
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal_tma<128, 1><<<gsp, THREADS, kLseSmemTma1>>>(
              qmap_lse, kmap_lse, d_lse, d_lse_part, S, H, Hkv, scale, lse_split_eff);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows,
                                                       lse_split_eff);
        } else {
          lse_mma_kernel_bal_tma<128, 1><<<lg_bal, THREADS, kLseSmemTma1>>>(
              qmap_lse, kmap_lse, d_lse, nullptr, S, H, Hkv, scale, 1);
        }
      } else
#endif
      if (causal && lse_wgm) {
        if (lse_split_eff > 1) {
          // O40：K 维切片 + 二次归约（对齐 O38/O39 的 mma/TMA LSE）。
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal_wgmma<128, 1><<<gsp, THREADS, kLseSmemWgm1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale, nullptr, d_lse_part, lse_split_eff);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale);
        }
      } else if (causal) {
        if (lse_split_eff > 1) {
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal<128, 1><<<gsp, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale, nullptr, d_lse_part, lse_split_eff);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal<128, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                        Hkv, scale);
        }
      } else
        lse_mma_kernel<128><<<lg, THREADS, kLseSmem>>>(d_q, d_k, d_lse, S, H, Hkv, scale,
                                                       (int)causal);
      if (delta_warp_sel)
        delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      else
        delta_kernel<128><<<pg, THREADS>>>(d_o, d_do, d_delta, S, H);
    }
  };

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  auto run_all = [&]() {
    // O13：HD=128（NDT==1）时 dQ 由主 kernel 覆盖写，无需清零；只有 MLA（HD=512）走 RMW 累加才 memset。
    if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
    run_pre();
    run_main();
    // O24：dq_direct 时主 kernel 已直接写 bf16 dq ⇒ convert 跳过 dQ（n_q=0）。
    convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, dq, dk, dv,
                                                dq_direct ? 0 : n, nkv);
  };
  for (int i = 0; i < 3; ++i) run_all();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_all();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  ms /= iters;
  double flops = 4.0 * (double)B * S * H * S * D;
  double tflops = flops / (ms * 1e-3) / 1e12;
  printf("[timing] total(3 kernels) %.4f ms  %.2f TFLOPS (bwd FLOPs=4BS^2HD)\n", ms, tflops);

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_pre();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_pre = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev0, ev1));
  ms_pre /= iters;

  if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_main();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_main = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_main, ev0, ev1));
  ms_main /= iters;
  printf("[timing] preprocess %.4f ms | main %.4f ms | convert %.4f ms\n", ms_pre, ms_main,
         ms - ms_pre - ms_main);

  // ---- O24 A/B：delta 旧 block-per-row(smem 归约) vs 新 warp-per-row(向量化) ----
  if (D == 128) {
    auto time_delta = [&](int mode, float* out_ms) {
      auto launch = [&]() {
        if (mode)
          delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
        else
          delta_kernel<128><<<pg, THREADS>>>(d_o, d_do, d_delta, S, H);
      };
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float d_old = 0.f, d_new = 0.f;
    time_delta(0, &d_old);
    time_delta(1, &d_new);
    printf("[O24 A/B] delta old %.4f ms | warp %.4f ms (%.2fx)\n", d_old, d_new,
           d_old / d_new);
  }

  // ---- O8b A/B（仅 HD=128 且 causal）：LSE 原版(O8) vs 镜像配对 vs 镜像配对+cp.async ----
  if (D == 128 && causal) {
    auto time_lse = [&](int mode, float* out_ms) {
      auto launch = [&]() {
        if (mode == 3)
          lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(d_q, d_k, d_lse,
                                                                              S, H, Hkv, scale);
        else if (mode == 2)
          lse_mma_kernel_bal<128, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S,
                                                                        H, Hkv, scale);
        else if (mode == 1)
          lse_mma_kernel_bal<128, 0><<<lg_bal, THREADS, kLseSmemBal0>>>(d_q, d_k, d_lse, S,
                                                                        H, Hkv, scale);
        else
          lse_mma_kernel<128><<<lg, THREADS, kLseSmem>>>(d_q, d_k, d_lse, S, H, Hkv, scale, 1);
      };
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_o8 = 0.f, ms_bal = 0.f, ms_balp = 0.f;
    time_lse(0, &ms_o8);
    time_lse(1, &ms_bal);
    time_lse(2, &ms_balp);
    printf("[O8b A/B] lse O8 %.4f ms | bal(单缓冲) %.4f ms (%.3fx) | bal+cpasync(双缓冲) %.4f ms "
           "(%.3fx)\n", ms_o8, ms_bal, ms_o8 / ms_bal, ms_balp, ms_o8 / ms_balp);
    // 仅当显式开启 wgmma 时才跑 O9 A/B（sm_90 构建下 wgmma 是空实现，时序无意义）。
    if (lse_wgm) {
      float ms_wgm = 0.f;
      time_lse(3, &ms_wgm);
      printf("[O9 A/B] lse wgmma+SW128 %.4f ms (vs bal+cpasync %.3fx, vs O8 %.3fx)\n", ms_wgm,
             ms_balp / ms_wgm, ms_o8 / ms_wgm);
    }
#if defined(FA_WGMMA) && defined(FA_TMA)
    // ---- O31 A/B：LSE 的 wgmma+cp.async 版 vs 4D-TMA 版（同 session + 数值对拍）----
    if (lse_tma) {
      float* d_lse2 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lse2, (size_t)B * S * H * sizeof(float)));
      auto launch_w = [&]() {
        lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(d_q, d_k, d_lse2,
                                                                            S, H, Hkv, scale);
      };
      auto launch_t = [&]() {
        lse_mma_kernel_bal_tma<128, 1><<<lg_bal, THREADS, kLseSmemTma1>>>(
            qmap_lse, kmap_lse, d_lse, nullptr, S, H, Hkv, scale, 1);
      };
      auto time_one = [&](auto launch, float* out_ms) {
        for (int i = 0; i < 3; ++i) launch();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) launch();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        *out_ms = t / iters;
      };
      float ms_w = 0.f, ms_t = 0.f;
      time_one(launch_w, &ms_w);
      time_one(launch_t, &ms_t);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> l1((size_t)B * S * H), l2((size_t)B * S * H);
      CUDA_CHECK(cudaMemcpy(l1.data(), d_lse, l1.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(l2.data(), d_lse2, l2.size() * 4, cudaMemcpyDeviceToHost));
      double e = 0;
      for (size_t i = 0; i < l1.size(); ++i) e = std::max(e, (double)fabs(l1[i] - l2[i]));
      printf("[O31 A/B] lse tma %.4f ms | wgmma %.4f ms (wgmma/tma %.3fx) | "
             "max_abs(tma-vs-wgmma)=%.3e\n", ms_t, ms_w, ms_w / ms_t, e);
      cudaFree(d_lse2);
    }
    // ---- O38 A/B（bf16）：LSE 的 K 维 split + 二次归约（TMA 版，D=128/causal）----
    //   split=1 逐位退回 O31；split=2/4/8 各扫 1/split 的 K tile 切片后 merge。逐元素对拍
    //   以 split=1 为基准（理论等价、只差 fp32 求和次序）。
    if (lse_tma) {
      std::vector<float> ref_l((size_t)B * S * H, 0.f);
      float base_ms = 0.f, best = 1e9f;
      int bestk = 1;
      for (int sp = 1; sp <= 8; sp *= 2) {
        auto launch_sp = [&]() {
          if (sp == 1) {
            lse_mma_kernel_bal_tma<128, 1><<<lg_bal, THREADS, kLseSmemTma1>>>(
                qmap_lse, kmap_lse, d_lse, nullptr, S, H, Hkv, scale, 1);
          } else {
            dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * sp));
            lse_mma_kernel_bal_tma<128, 1><<<gsp, THREADS, kLseSmemTma1>>>(
                qmap_lse, kmap_lse, d_lse, d_lse_part, S, H, Hkv, scale, sp);
            const long long nrows = (long long)B * S * H;
            const int th = 256;
            const long long bl = (nrows + th - 1) / th;
            lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, sp);
          }
        };
        for (int i = 0; i < 3; ++i) launch_sp();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) launch_sp();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        t /= iters;
        std::vector<float> got((size_t)B * S * H);
        CUDA_CHECK(cudaMemcpy(got.data(), d_lse, got.size() * 4, cudaMemcpyDeviceToHost));
        double e = 0.0;
        if (sp == 1) {
          ref_l = got;
          base_ms = t;
        } else {
          for (size_t i = 0; i < got.size(); ++i)
            e = std::max(e, (double)std::fabs((double)got[i] - (double)ref_l[i]));
        }
        printf("[O38 A/B] lse split=%d %.4f ms (%.3fx vs split1) | max_abs vs split1=%.3e\n",
               sp, t, base_ms / t, e);
        if (t < best) { best = t; bestk = sp; }
      }
      printf("[O38] best lse split=%d %.4f ms (%.3fx)\n", bestk, best, base_ms / best);
    }
#endif
  }

  // ---- O39 A/B（D=512/MLA/causal）：`lse_mma_kernel_bal<512>` 的 K 维 split + 二次归约 ----
  //   split=1 逐位退回 O8b 原路径；split>1 各扫 1/split 的 K tile 切片后 merge。逐元素对拍
  //   以 split=1 为基准（理论等价、只差 fp32 求和次序）。仅 causal（非 causal 走 O8 原版）。
  if (causal && D == 512) {
    std::vector<float> ref_l((size_t)B * S * H, 0.f);
    float base_ms = 0.f, best = 1e9f;
    int bestk = 1;
    for (int sp = 1; sp <= 16; sp *= 2) {
      auto launch_sp = [&]() {
        if (sp == 1) {
          lse_mma_kernel_bal<512, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                        Hkv, scale);
        } else {
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * sp));
          lse_mma_kernel_bal<512, 1><<<gsp, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale, nullptr, d_lse_part, sp);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, sp);
        }
      };
      for (int i = 0; i < 3; ++i) launch_sp();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_sp();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      t /= iters;
      std::vector<float> got((size_t)B * S * H);
      CUDA_CHECK(cudaMemcpy(got.data(), d_lse, got.size() * 4, cudaMemcpyDeviceToHost));
      double e = 0.0;
      if (sp == 1) { ref_l = got; base_ms = t; }
      else
        for (size_t i = 0; i < got.size(); ++i)
          e = std::max(e, (double)std::fabs((double)got[i] - (double)ref_l[i]));
      printf("[O39 A/B] lse(D=512) split=%d %.4f ms (%.3fx vs split1) | max_abs vs split1=%.3e\n",
             sp, t, base_ms / t, e);
      if (t < best) { best = t; bestk = sp; }
    }
    printf("[O39] best lse split=%d %.4f ms (%.3fx)\n", bestk, best, base_ms / best);
  }

  double main_flops = 4.0 * (double)B * S * H * S * D;
  // ---- O6 A/B（仅 HD=128）：同 session 对比原版（PIPE=0）、K/V 双缓冲（PIPE=1）、只 K 双缓冲（PIPE=2）----
  if (D == 128) {
  auto launch_mode = [&](int m) {
    if (m == 2)
      launch_bwd_mma<128, 64, 32, 2>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    else if (m == 1)
      launch_bwd_mma<128, 64, 32, 1>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    else
      launch_bwd_mma<128, 64, 32, 0>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
  };
  auto time_launch = [&](int m, float* out_ms) {
    if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
    for (int i = 0; i < 3; ++i) launch_mode(m);
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) launch_mode(m);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float t = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
    *out_ms = t / iters;
  };
  float ms_nopipe = 0.f, ms_pipe = 0.f, ms_pipe2 = 0.f;
  time_launch(0, &ms_nopipe);
  time_launch(1, &ms_pipe);
  time_launch(2, &ms_pipe2);
  printf("[O6 A/B] main nopipe %.4f ms (%.2f TF) | pipe(KV both) %.4f ms (%.2f TF) | "
         "pipe2(K only) %.4f ms (%.2f TF) => nopipe/pipe %.3fx, nopipe/pipe2 %.3fx\n",
         ms_nopipe, main_flops / (ms_nopipe * 1e-3) / 1e12, ms_pipe,
         main_flops / (ms_pipe * 1e-3) / 1e12, ms_pipe2,
         main_flops / (ms_pipe2 * 1e-3) / 1e12, ms_nopipe / ms_pipe,
         ms_nopipe / ms_pipe2);

  // ---- O6c A/B（仅 causal，仍在 HD=128 块内）：mblk 重排 sched=0/1/2（同一 pipe）----
  if (causal) {
    const int mdef = (pipe < 0) ? auto_pipe : pipe;
    float ms_s[3] = {0.f, 0.f, 0.f};
    for (int s = 0; s < 3; ++s) {
      sched = s;
      time_launch(mdef, &ms_s[s]);
    }
    printf("[O6c A/B] main sched0(原样) %.4f ms | sched1(交错) %.4f ms (%.3fx) | "
           "sched2(逆序) %.4f ms (%.3fx)\n",
           ms_s[0], ms_s[1], ms_s[0] / ms_s[1], ms_s[2], ms_s[0] / ms_s[2]);
    sched = 1;
  }

  // ---- O9b A/B（仅 FA_WGMMA 且 causal）：主 kernel mma(64,64,2) vs wgmma(GEMM1/2 用 wgmma) ----
#ifdef FA_WGMMA
  if (causal) {
    auto time_wg = [&](bool use_wg, float* out_ms) {
      auto launch = [&]() {
        if (use_wg)
          launch_bwd_wgmma<128>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                d_dk_acc, d_dv_acc, S, H, Hkv, scale, 1, 0);
        else
          launch_cfg(64, 64, 2, false, true);
      };
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_mma = 0.f, ms_wg = 0.f;
    time_wg(false, &ms_mma);
    time_wg(true, &ms_wg);
    auto tf = [&](float ms) { return main_flops / (ms * 1e-3) / 1e12; };
    printf("[O9b A/B] main mma(64,64,2) %.4f ms (%.2f TF) | wgmma(GEMM1/2) %.4f ms "
           "(%.2f TF) => %.3fx\n",
           ms_mma, tf(ms_mma), ms_wg, tf(ms_wg), ms_mma / ms_wg);
  }

  // ---- O17 A/B（仅 FA_WGMMA 构建，HD=128）：mma 最优档 vs O9b wgmma vs O17 wgmma2(BM=128) ----
  {
    auto time_o17 = [&](int mode, float* out_ms, float* dq_c, float* dk_c, float* dv_c) {
      auto launch = [&]() {
        if (mode == 2) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 5) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 6) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2b<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 7) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2b<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                         d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
#if defined(FA_TMA)
        } else if (mode == 8) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2b_tma<128, true>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                            d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                            (int)causal);
        } else if (mode == 9) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2b_tma<128, false>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                             d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                             (int)causal);
        } else if (mode == 10) {
          // O36：BN=64 wgmma2 的逐 atom 4D-TMA（对齐 fp16 O35）。
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2_tma<128, true>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                           d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                           (int)causal);
#endif
        } else if (mode == 1) {
          dim3 g((S + 63) / 64, H, B);
          launch_bwd_wgmma<128>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc,
                                d_dv_acc, S, H, Hkv, scale, (int)causal, 0);
        } else {
          launch_cfg(64, 64, 2, false, true);
        }
      };
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
      if (dq_c) CUDA_CHECK(cudaMemcpy(dq_c, d_dq_acc, n * sizeof(float), cudaMemcpyDeviceToHost));
      if (dk_c) CUDA_CHECK(cudaMemcpy(dk_c, d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
      if (dv_c) CUDA_CHECK(cudaMemcpy(dv_c, d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    };
    std::vector<float> q0(n), k0(nkv), v0(nkv), q1(n), k1(nkv), v1(nkv), q2(n), k2(nkv),
        v2(nkv), q5(n), k5(nkv), v5(nkv), q6(n), k6(nkv), v6(nkv), q7(n), k7(nkv), v7(nkv);
#if defined(FA_TMA)
    std::vector<float> q8(n), k8(nkv), v8(nkv), q10(n), k10(nkv), v10(nkv);
#endif
    float ms_mma = 0.f, ms_wgm = 0.f, ms_wg2 = 0.f, ms_wg2ns = 0.f, ms_wg2b = 0.f, ms_wg2bns = 0.f;
    time_o17(0, &ms_mma, q0.data(), k0.data(), v0.data());
    time_o17(1, &ms_wgm, q1.data(), k1.data(), v1.data());
    time_o17(2, &ms_wg2, q2.data(), k2.data(), v2.data());
    time_o17(5, &ms_wg2ns, q5.data(), k5.data(), v5.data());
    time_o17(6, &ms_wg2b, q6.data(), k6.data(), v6.data());
    time_o17(7, &ms_wg2bns, q7.data(), k7.data(), v7.data());
#if defined(FA_TMA)
    float ms_wg2btma = 0.f, ms_wg2tma = 0.f;
    time_o17(8, &ms_wg2btma, q8.data(), k8.data(), v8.data());
    time_o17(10, &ms_wg2tma, q10.data(), k10.data(), v10.data());
#endif
    auto mad = [](const std::vector<float>& a, const std::vector<float>& b) {
      double d = 0; for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i]-b[i])); return d;
    };
    printf("[O17 A/B] main mma(64,64,2) %.4f ms | O9b wgmma %.4f (%.3fx) | O17 wgmma2(BM128) "
           "%.4f (%.3fx) TF %.1f\n",
           ms_mma, ms_wgm, ms_mma / ms_wgm, ms_wg2, ms_mma / ms_wg2,
           main_flops / (ms_wg2 * 1e-3) / 1e12);
    printf("[O17-2 A/B] main O17 wg2 wg0串行(dV+dK) %.4f ms (%.1f TF) | O17-2 拆分(wg0=dV,"
           "wg1=dK) %.4f ms (%.1f TF) => %.3fx | max|diff| dq/dk/dv=%.2e/%.2e/%.2e\n",
           ms_wg2ns, main_flops / (ms_wg2ns * 1e-3) / 1e12, ms_wg2,
           main_flops / (ms_wg2 * 1e-3) / 1e12, ms_wg2ns / ms_wg2, mad(q2, q5), mad(k2, k5),
           mad(v2, v5));
    printf("[O17 A/B] max|diff| wg2-vs-mma dq/dk/dv=%.2e/%.2e/%.2e | wg2-vs-O9b "
           "dq/dk/dv=%.2e/%.2e/%.2e\n",
           mad(q2, q0), mad(k2, k0), mad(v2, v0), mad(q2, q1), mad(k2, k1), mad(v2, v1));
    printf("[O18 A/B] main O17 wg2(BN64) %.4f ms (%.1f TF) | O18 wg2b(BN128,split) %.4f ms "
           "(%.1f TF) => %.3fx | O18 wg2b(BN128,串行) %.4f ms => %.3fx | "
           "max|diff| wg2b-vs-wg2 dq/dk/dv=%.2e/%.2e/%.2e | wg2bns-vs-wg2b %.2e/%.2e/%.2e\n",
           ms_wg2, main_flops / (ms_wg2 * 1e-3) / 1e12, ms_wg2b,
           main_flops / (ms_wg2b * 1e-3) / 1e12, ms_wg2 / ms_wg2b, ms_wg2bns,
           ms_wg2 / ms_wg2bns, mad(q6, q2), mad(k6, k2), mad(v6, v2), mad(q7, q6), mad(k7, k6),
           mad(v7, v6));
#if defined(FA_TMA)
    printf("[O34 A/B] main O18 wg2b(cp.async) %.4f ms (%.1f TF) | O34 wg2b(TMA,split) %.4f ms "
           "(%.1f TF) => %.3fx | max|diff| tma-vs-cpasync dq/dk/dv=%.2e/%.2e/%.2e\n",
            ms_wg2b, main_flops / (ms_wg2b * 1e-3) / 1e12, ms_wg2btma,
            main_flops / (ms_wg2btma * 1e-3) / 1e12, ms_wg2b / ms_wg2btma, mad(q8, q6),
            mad(k8, k6), mad(v8, v6));
    printf("[O36 A/B] main O17 wg2(BN64,cp.async) %.4f ms (%.1f TF) | O36 wg2(BN64,TMA,split) "
           "%.4f ms (%.1f TF) => %.3fx | max|diff| tma-vs-cpasync dq/dk/dv=%.2e/%.2e/%.2e\n",
           ms_wg2, main_flops / (ms_wg2 * 1e-3) / 1e12, ms_wg2tma,
           main_flops / (ms_wg2tma * 1e-3) / 1e12, ms_wg2 / ms_wg2tma, mad(q10, q2),
           mad(k10, k2), mad(v10, v2));
#endif
  }
#endif
  }  // end if (D == 128)

  // ---- O5c A/B（仅 HD=128 且 causal）：不同 (BM,BN,PIPE) tile 配置 ----
  if (D == 128 && causal) {
    auto time_cfg = [&](int bm, int bn, int pp, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, false, true); };
      if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms0 = 0.f, ms1 = 0.f, ms2 = 0.f, ms3 = 0.f;
    time_cfg(64, 32, 2, &ms0);
    time_cfg(64, 64, 2, &ms1);
    time_cfg(32, 32, 1, &ms2);
    time_cfg(32, 32, 2, &ms3);
    printf("[O5c A/B] main (64,32,2) %.4f ms (%.2f TF) | (64,64,2) %.4f (%.2f) | (32,32,1) %.4f "
           "(%.2f) | (32,32,2) %.4f (%.2f) => best %.3fx\n",
           ms0, main_flops / (ms0 * 1e-3) / 1e12, ms1, main_flops / (ms1 * 1e-3) / 1e12,
           ms2, main_flops / (ms2 * 1e-3) / 1e12, ms3, main_flops / (ms3 * 1e-3) / 1e12,
           ms0 / std::min(std::min(ms0, ms1), std::min(ms2, ms3)));
  }

  // ---- O7c A/B（仅 HD=128 且 causal）：4 个几何 × {float2/float4} × {无/有 LSE-D 预装} ----
  if (D == 128 && causal) {
    auto time_r4 = [&](int bm, int bn, int pp, bool r4, bool prel, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, r4, prel); };
      if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    const int cfgs[4][3] = {{64, 32, 2}, {64, 64, 2}, {32, 32, 1}, {32, 32, 2}};
    printf("[O7c A/B] main-only: base(f2,noPREL) | f4 | f2+PREL | f4+PREL, main-only:\n");
    for (int ci = 0; ci < 4; ++ci) {
      const int bm = cfgs[ci][0], bn = cfgs[ci][1], pp = cfgs[ci][2];
      float a = 0.f, b = 0.f, c = 0.f, e = 0.f;
      time_r4(bm, bn, pp, false, false, &a);
      time_r4(bm, bn, pp, true, false, &b);
      time_r4(bm, bn, pp, false, true, &c);
      time_r4(bm, bn, pp, true, true, &e);
      auto tf = [&](float ms) { return main_flops / (ms * 1e-3) / 1e12; };
      printf("  (BM=%d,BN=%d,PIPE=%d): base %.4f (%.2f) | f4 %.4f (%.2f,%+.1f%%) | "
             "f2+PREL %.4f (%.2f,%+.1f%%) | f4+PREL %.4f (%.2f,%+.1f%%)\n",
             bm, bn, pp, a, tf(a), b, tf(b), (a / b - 1) * 100, c, tf(c), (a / c - 1) * 100, e,
             tf(e), (a / e - 1) * 100);
    }
  }

  // ---- MLA（HD=512）配置 A/B：BM=32/64 × PIPE=0/1 ----
  if (D == 512 && causal) {
    auto time_cfg2 = [&](int bm, int bn, int pp, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, false, true); };
      if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float a = 0.f, b = 0.f, c = 0.f;
    time_cfg2(32, 32, 0, &a);
    time_cfg2(32, 32, 1, &b);
    time_cfg2(64, 32, 0, &c);
    auto tf = [&](float ms) { return main_flops / (ms * 1e-3) / 1e12; };
    printf("[MLA512 A/B] main: (32,32,0) %.4f (%.2f TF) | (32,32,1) %.4f (%.2f) | "
           "(64,32,0) %.4f (%.2f) => best %.3fx\n",
           a, tf(a), b, tf(b), c, tf(c), a / std::min(a, std::min(b, c)));
  }

  // ---- 数值对拍（重新跑一次完整 forward 保证累加缓冲清零）----
  run_all();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<bf16> hdq(n), hdk(nkv), hdv(nkv);
  CUDA_CHECK(cudaMemcpy(hdq.data(), dq, n * sizeof(bf16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdk.data(), dk, nkv * sizeof(bf16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdv.data(), dv, nkv * sizeof(bf16), cudaMemcpyDeviceToHost));
  std::vector<float> mdq(n), mdk(nkv), mdv(nkv);
  for (size_t i = 0; i < n; ++i) mdq[i] = __bfloat162float(hdq[i]);
  for (size_t i = 0; i < nkv; ++i) {
    mdk[i] = __bfloat162float(hdk[i]);
    mdv[i] = __bfloat162float(hdv[i]);
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
