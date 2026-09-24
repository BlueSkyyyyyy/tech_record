// =============================================================================
// fa_bwd_fp16_mma_main.cu —— fp16 张量核反向（O5）**两文件版的 host 部分**
// =============================================================================
// device 代码（mma/ldmatrix 封装、preprocess_kernel、fa_bwd_fp16_mma_kernel、
// convert_kernel）见 `fa_bwd_fp16_mma_kernels.cuh`；本文件只保留 host 侧：
// npy 读取 / launcher / 自测对拍。行为与单文件 `fa_bwd_fp16_mma_onefile.cu`
// **逐位一致**（device 代码逐字未改）。
//
// 用法：run.sh src/fp16/fa_bwd_fp16_mma_main.cu [--dir=...] [--full|--causal] [--iters=N]
// =============================================================================

#include "fa_bwd_fp16_mma_kernels.cuh"

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
// O30：为 LSE 的 Q/K 建 4D TMA 描述符（dims={D,S,H,B}，SW128，box={64,64,1,1}）。
// globalStride（字节）：dim1(S) 的行距 = H*D*2，dim2(H) 的头距 = D*2，dim3(B) 的批距。
// 要求 16B 对齐（D=128 时 256 的整数倍，恒成立）。坐标 {k0,row,head,batch}。
static CUtensorMap make_lse_map(const void* ptr, long long H, long long S, long long D,
                                long long B) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D * 2), (uint64_t)(D * 2),
                         (uint64_t)(S * H * D * 2)};
  uint32_t box[4] = {64, 64, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, (void*)ptr, dims, strides, box, estr,
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
static void launch_bwd_mma(dim3 mg, const __half* q, const __half* k, const __half* v,
                           const __half* do_, const float* delta, const float* lse,
                           float* dq_acc, float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                           float scale, int causal, int sched) {
  constexpr int kvn = (PIPE == 0) ? 2 : (PIPE == 1 ? 4 : 3);
  constexpr int pds = (PIPE == 2) ? 2 * BM * (BN + 8) : 2 * BN * (BM + 8) + BM * (BN + 8);
  constexpr int smem =
      (2 * BM * (HD + 8) + kvn * BN * (HD + 8) + pds) * (int)sizeof(__half);
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_mma_kernel<HD, BM, BN, PIPE, R4, PREL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_mma_kernel<HD, BM, BN, PIPE, R4, PREL><<<mg, THREADS, smem>>>(
      q, k, v, do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, sched);
}

#ifdef FA_WGMMA
// O9b/O9b-2：wgmma 主 kernel（只 HD=128 / BM=BN=64）。Q/dO/K/V SW128 + P/dS SW128，≈97KB。
template <int HD, bool OW = false>
static void launch_bwd_wgmma(dim3 mg, const __half* q, const __half* k, const __half* v,
                             const __half* do_, const float* delta, const float* lse,
                             float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                             int Hkv, float scale, int causal, int sched) {
  constexpr int BM = 64, BN = 64;
  constexpr int TILE  = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int smem = 1024 + TILE * 2 + KTILE * 3 + 2 * BM * BN * (int)sizeof(__half);
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma_kernel<HD, OW>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_wgmma_kernel<HD, OW><<<mg, THREADS, smem>>>(q, k, v, do_, delta, lse, dq_acc,
                                                          dk_acc, dv_acc, S, H, Hkv, scale,
                                                          causal, sched);
}

// O17：2 warpgroup（BM=128）wgmma 主 kernel（只 HD=128）。Q/dO/K/V + P/dS 全 SW128。
// smem = 1024(对齐) + Q 32KB + dO 32KB + K 2×16KB + V 16KB + P 16KB + dS 16KB ≈ 145KB。
template <int HD, bool SPLIT = true, int CL = 1>
static void launch_bwd_wgmma2(dim3 mg, const __half* q, const __half* k, const __half* v,
                              const __half* do_, const float* delta, const float* lse,
                              float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                              int Hkv, float scale, int causal,
                              __half* dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2 只做 HD=128");
  constexpr int BM = 128, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  // O25：CL>1 时 leader 的 smem 里多两个 [BN][HD] fp32 合并累加器（dK/dV）。
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2 +
                       (CL > 1 ? 2 * BN * HD * (int)sizeof(float) : 0);
  static_assert(CL == 1 || CL == 2, "cluster 只做 2");
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma2_kernel<HD, SPLIT, CL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if constexpr (CL > 1) {
    // O25：用 cudaLaunchKernelEx 指定 cluster 维度（沿 x，相邻两个 mblk 配对）。
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = mg;
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = nullptr;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, fa_bwd_fp16_wgmma2_kernel<HD, SPLIT, CL>, q, k, v,
                                  do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale,
                                  causal, dq_h));
  } else {
    fa_bwd_fp16_wgmma2_kernel<HD, SPLIT, CL><<<mg, 256, smem>>>(
        q, k, v, do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, dq_h);
  }
}

// O18：BN=128 版 wgmma2（只 HD=128）。tile 数减半；smem = 1024 + Q32 + dO32 + K 2×32 + V32 + P32 + dS32
// ≈ 230400B（仍 1 CTA/SM）。
template <int HD, bool SPLIT = true, bool DET = false>
static void launch_bwd_wgmma2b(dim3 mg, const __half* q, const __half* k, const __half* v,
                               const __half* do_, const float* delta, const float* lse,
                               float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                               int Hkv, float scale, int causal,
                               float* dk_part = nullptr, float* dv_part = nullptr,
                               int nblk = 0, __half* dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2b 只做 HD=128");
  constexpr int BM = 128, BN = 128;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma2b_kernel<HD, SPLIT, DET>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_wgmma2b_kernel<HD, SPLIT, DET><<<mg, 256, smem>>>(
      q, k, v, do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, dk_part,
      dv_part, nblk, dq_h);
}

// O17b：4 warpgroup（BM=256）wgmma 主 kernel（只 HD=128）。Q/dO/P/dS SW128，K/V 单缓冲 + 后段预取。
// smem = 1024(对齐) + Q 64KB + dO 64KB + K 16KB + V 16KB + P 32KB + dS 32KB = 224KB（1 CTA/SM）。
template <int HD, bool SEQ = false>
static void launch_bwd_wgmma4(dim3 mg, const __half* q, const __half* k, const __half* v,
                              const __half* do_, const float* delta, const float* lse,
                              float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                              int Hkv, float scale, int causal) {
  static_assert(HD == 128, "wgmma4 只做 HD=128");
  constexpr int BM = 256, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 2 + PTILE * 2;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma4_kernel<HD, SEQ>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_wgmma4_kernel<HD, SEQ><<<mg, 512, smem>>>(q, k, v, do_, delta, lse, dq_acc,
                                                        dk_acc, dv_acc, S, H, Hkv, scale, causal);
}
#endif

int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16";
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
  // O30：LSE 是否用 TMA 版（仅 FA_TMA 构建、D==128、causal；1=用 4D TMA 载入 Q/K）。
  //   -1=自动（FA_TMA 构建下 D==128/causal 默认开），0/1 由 `--lsetma=` 强制。
  int lse_tma = -1;
  // O9b：主 kernel 是否用 wgmma（仅 FA_WGMMA 构建、D==128 且 sel=(64,64) 时生效）。
  int wgmma_sel = 0;
  // O16：wgmma 主 kernel 是否用「分段 wait_group」重叠 epilogue（-1=自动/开，0=关，1=开）。
  int ow_opt = -1;
  // O17：2 warpgroup（BM=128，跨 wg 归约）wgmma 主 kernel（仅 FA_WGMMA 构建、D==128）。
  // O23：默认自动选择（见下方 auto 段）；`--wg2=0/1` 可强制。wg_forced=用户显式指定。
  int wg2_sel = 0;
  // O18：BN=128 版 wgmma2（仅 FA_WGMMA 构建、D==128）。O23：默认自动选择。
  int wg2bn_sel = 0;
  bool wg_forced = false;
  // O17-2：wgmma2 的 GEMM3/GEMM4 是否拆分到两个 warpgroup（1=拆，0=原版 wg0 串行）。
  int wg2split_sel = 1;
  // O17b：4 warpgroup（BM=256，跨 wg 归约再砍半）wgmma 主 kernel（仅 FA_WGMMA 构建、D==128）。
  int wg4_sel = 0;
  // O17b：是否用「串行 GEMM1/2 + 读回 P」版（消 spill）；-1=自动（SEQ=1）。
  int wg4seq_opt = -1;
  // O7b：是否跑「确定性 dK/dV（partial + 二次归约）」A/B（仅 FA_WGMMA 构建、D==128）。
  int det_ab = 0;
  // O24：delta 用 warp-per-row 向量化版（1，默认）还是旧 block-per-row smem 版（0，A/B）。
  int delta_warp_sel = 1;
  // O24：D==128 wgmma2/2b 路径直接用 fp16 写 dQ、convert 跳过 dQ（1，默认；0=A/B）。
  int dq_direct_sel = 1;
  // O25：cluster 分布式归约（仅 HD=128、BN=64 的 wgmma2；cluster 沿 bx 配对相邻 mblk）。
  //   0=关（默认），2=开。开了会强制走 wgmma2(BN=64)（BN=128 的 2b 放不下合并累加器）。
  int cluster_sel = 0;
  int iters = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--full") causal = false;
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
    else if (a.rfind("--wgmma=", 0) == 0) wgmma_sel = atoi(a.c_str() + 8);
    else if (a == "--wgmma") wgmma_sel = 1;
    else if (a.rfind("--ow=", 0) == 0) ow_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--wg2=", 0) == 0) { wg2_sel = atoi(a.c_str() + 6); wg_forced = true; }
    else if (a.rfind("--wg2split=", 0) == 0) wg2split_sel = atoi(a.c_str() + 11);
    else if (a == "--wg2") { wg2_sel = 1; wg_forced = true; }
    else if (a.rfind("--wg2bn=", 0) == 0) { wg2bn_sel = atoi(a.c_str() + 8); wg_forced = true; }
    else if (a == "--wg2bn") { wg2bn_sel = 1; wg_forced = true; }
    else if (a.rfind("--wg4=", 0) == 0) wg4_sel = atoi(a.c_str() + 6);
    else if (a == "--wg4") wg4_sel = 1;
    else if (a.rfind("--wg4seq=", 0) == 0) wg4seq_opt = atoi(a.c_str() + 9);
    else if (a.rfind("--det=", 0) == 0) det_ab = atoi(a.c_str() + 6);
    else if (a == "--det") det_ab = 1;
    else if (a.rfind("--deltawarp=", 0) == 0) delta_warp_sel = atoi(a.c_str() + 12);
    else if (a.rfind("--dqdirect=", 0) == 0) dq_direct_sel = atoi(a.c_str() + 11);
    else if (a.rfind("--cluster=", 0) == 0) cluster_sel = atoi(a.c_str() + 10);
    else if (a == "--cluster") cluster_sel = 2;
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
  const int Hkv = (int)k_np.shape[2];
  if (D != 128 && D != 512) {
    fprintf(stderr, "O5 fp16 mma 版支持 head_dim=128/512；当前 %d\n", D);
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
    std::vector<__half> h(src.size());
    for (size_t i = 0; i < src.size(); ++i) h[i] = __float2half(src[i]);
    return h;
  };
  auto qh = to_half(q_np.data), kh = to_half(k_np.data), vh = to_half(v_np.data),
       doh = to_half(do_np.data), oh = to_half(o_np.data);

  __half *dq, *dk, *dv;
  __half *d_q, *d_k, *d_v, *d_o, *d_do;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc;
  CUDA_CHECK(cudaMalloc(&dq, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dk, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dv, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_k, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_v, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_o, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_do, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_delta, (size_t)B * S * H * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, (size_t)B * S * H * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, nkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, nkv * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, qh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, kh.data(), nkv * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, vh.data(), nkv * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o, oh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do, doh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

  dim3 pg(S, H, B);
  dim3 mg((S + 63) / 64, H, B);
  const int lse_nblk = (S + LBM - 1) / LBM;
  dim3 lg(lse_nblk, H, B);
  dim3 lg_bal((lse_nblk + 1) / 2, H, B);   // O8b：镜像配对，grid.x 减半
  // LSE smem 随 head_dim 变化：Qs[LBM*LD] +（PIPE=0 时 1 份 / PIPE=1 时 2 份）Ks[LBN*LD]，LD=HD+8。
  const int LDl = D + 8;
  const int kLseSmem = (LBM + LBN) * LDl * (int)sizeof(__half);
  // O8b：PIPE=0 单缓冲（与 O8 同尺寸），PIPE=1 双缓冲。
  const int kLseSmemBal0 = (LBM + LBN) * LDl * (int)sizeof(__half);
  const int kLseSmemBal1 = (LBM + 2 * LBN) * LDl * (int)sizeof(__half);
  // O9：wgmma LSE（仅 D=128）用 SW128 tile：TILE=(LBM/8)*(D/64)*1024 B，另加 1024B 对齐余量。
  const int kLseTileWgm = (LBM / 8) * (D / 64) * 1024;
  const int kLseSmemWgm0 = 1024 + kLseTileWgm * 2;  // Q + K（单缓冲）
  const int kLseSmemWgm1 = 1024 + kLseTileWgm * 3;  // Q + 2×K
  // O30：TMA 版与 wgmma 版同布局（Q + 2×K）+ 3 个 mbarrier（24B，取 64B 余量）。
  const int kLseSmemTma1 = 1024 + kLseTileWgm * 3 + 64;
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
  // O30：建 LSE 的 Q/K 4D TMA 描述符（一次，供所有 (h,b) CTA 用坐标选择）。
  CUtensorMap qmap_lse, kmap_lse;
  if (D == 128) {
    qmap_lse = make_lse_map(d_q, H, S, D, B);
    kmap_lse = make_lse_map(d_k, Hkv, S, D, B);
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
  // O13：auto tile 重新标定（O7c-PREL 之后）。
  //  O6c 时代 `grid<132 且 S≤1024` 用 (BM=32,BN=32,PIPE=1) 换并行度；但 O7c-PREL（LSE/D
  //  预装寄存器）与其它优化后，BM=64 的每 CTA 效率更高：实测 S=512 MHA 主 kernel
  //  (64,64,2) 比 (32,32,1) 快 1.23×（O5c A/B），端到端 1.09×。故取消 BM=32 分支。
  //  BN=64 能降 L1/TEX，但 smem 105KB→2 CTA/SM；当 grid 落在「2 个波」的坏量化点
  //  （256<grid≤600，实测 grid=512 的 S1024/GQA-kv4）时反而输给 BN=32（3 CTA/SM），
  //  故仅该区间保留 BN=32。
  const long long grid = (long long)((S + 63) / 64) * H * B;
  const bool bn64 = (S >= 4096) || (grid <= 256) || (grid > 600);
  int auto_bm = 64;
  int auto_bn = bn64 ? 64 : 32;
  int auto_pipe = (grid >= 396) ? 2 : 1;
  if (D == 512) {
    // MLA（HD=512）：K/V 行有 520 个 half，BM=64 时双缓冲会超 smem，BM=32 时 PIPE=1
    // （K 双缓冲）仍在 232KB 内且实测最快（1.67× vs PIPE=0）；S/H 小、grid 不足一个波。
    auto_bm = 32;
    auto_bn = 32;
    auto_pipe = 1;
  }
  const int bm_sel = (bm_opt > 0) ? bm_opt : auto_bm;
  const int bn_sel = (bn_opt > 0) ? bn_opt : auto_bn;
  const int pp_sel = (pipe >= 0) ? pipe : auto_pipe;
  printf("[O6c] main grid=%lld auto=(BM=%d,BN=%d,PIPE=%d) sel=(BM=%d,BN=%d,PIPE=%d)\n", grid,
         auto_bm, auto_bn, auto_pipe, bm_sel, bn_sel, pp_sel);
  // O7c：`r4`（float4 归约）与 `prel`（LSE/D 预装寄存器）为运行期开关，各自派发到两个
  // 模板实例，便于同一 session 内做 2×2 A/B。
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
      // MLA：BN=32；BM 可 32/64；只支持 PIPE=0/1（PIPE=2 的 V 单缓冲无意义且更费 smem）。
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
  // O16：默认关（实测中性 1.004×，且会改 dK/dV 的 atomic 次序、破坏历史逐位值）；保留 A/B。
  const bool ow_sel = (ow_opt > 0);
  // O17b：SEQ（串行 GEMM1/2 + 读回 half P）默认关（有 fp16 精度损失）；默认走合并版。
  const bool wg4seq_sel = (wg4seq_opt > 0);
  // O23：把 O17/O18 的 Hopper wgmma2 主 kernel 在 `-DFA_WGMMA`（sm_90a）构建下**默认打开**
  //   （对齐 fp8 的 O22）。仅 D==128：S>=4096 用 BN=128（O18，tile 数减半），否则 BN=64（O17）。
  //   用户显式传 `--wg2=`/`--wg2bn=` 时不做自动选择；`--wg2=0 --wg2bn=0` 退回 mma 路径做 A/B。
  //   非 FA_WGMMA 构建行为完全不变（wg2_sel/wg2bn_sel 恒 0）。
#ifdef FA_WGMMA
  if (!wg_forced && D == 128) {
    if (S >= 4096) wg2bn_sel = 1; else wg2_sel = 1;
  }
  // O25：cluster 版只存在于 BN=64 的 wgmma2（BN=128 的 2b 放不下 [BN][HD]×2 合并累加器），
  //   故 --cluster 开启时强制走 wgmma2；grid.x=nblk 须能被 cluster 整除，否则退回普通 wg2。
  int cluster_use = 0;
  if (cluster_sel > 1 && D == 128) {
    const int nblk = (S + 127) / 128;
    if (nblk % cluster_sel == 0) { wg2_sel = 1; wg2bn_sel = 0; cluster_use = cluster_sel; }
  }
  // O23：LSE 预处理也默认走 Hopper wgmma 版（仅 causal / D==128；非 causal 自动落回 O8 原版）。
  if (!lse_forced && D == 128 && causal) lse_wgm = 1;
#endif
  // O30：TMA 版 LSE 需驱动 API（`cuTensorMapEncodeTiled`）⇒ 只有 `-DFA_TMA -lcuda` 构建才编译
  //   该路径；此时 D==128/causal 默认开（1.30–1.36× 于 wgmma+cp.async，且逐位相同）。
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (lse_tma < 0) lse_tma = (D == 128 && causal) ? 1 : 0;
#else
  if (lse_tma < 0) lse_tma = 0;
#endif
  printf("[O23] main backend = %s | lse = %s (D=%d S=%d)%s\n",
         wg2bn_sel ? "wgmma2b(BN=128)" : (wg2_sel ? "wgmma2(BN=64)" : "mma"),
         (D == 128 && causal && lse_wgm) ? "wgmma" : "mma", D, S,
#ifdef FA_WGMMA
         cluster_use ? " +cluster2" : ""
#else
         ""
#endif
  );
  // O24：D==128 的 wgmma2/wgmma2b 路径里 dQ 唯一拥有 ⇒ 主 kernel 直接写 fp16 `dq`，
  // `convert_kernel` 跳过 dQ（n_q 传 0）。其它路径（mma/wgmma/wgmma4/MLA）仍写 fp32 dq_acc。
  bool dq_direct = false;
  auto run_main = [&]() {
#ifdef FA_WGMMA
    if (wg4_sel && D == 128) {
      dim3 g((S + 255) / 256, H, B);
      if (wg4seq_sel)
        launch_bwd_wgmma4<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
      else
        launch_bwd_wgmma4<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
      return;
    }
    if (wg2bn_sel && D == 128) {
      dim3 g((S + 127) / 128, H, B);
      dq_direct = dq_direct_sel;
      __half* dqo = dq_direct_sel ? dq : nullptr;
      if (wg2split_sel)
        launch_bwd_wgmma2b<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, nullptr,
                                      nullptr, 0, dqo);
      else
        launch_bwd_wgmma2b<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, nullptr,
                                       nullptr, 0, dqo);
      return;
    }
    if (wg2_sel && D == 128) {
      dim3 g((S + 127) / 128, H, B);
      dq_direct = dq_direct_sel;
      __half* dqo = dq_direct_sel ? dq : nullptr;
      if (cluster_use == 2) {
        // O25：cluster 分布式归约（仅 SPLIT 版）。
        launch_bwd_wgmma2<128, true, 2>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo);
        return;
      }
      if (wg2split_sel)
        launch_bwd_wgmma2<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo);
      else
        launch_bwd_wgmma2<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo);
      return;
    }
    if (wgmma_sel && D == 128 && bm_sel == 64 && bn_sel == 64) {
      dim3 g((S + 63) / 64, H, B);
      if (ow_sel)
        launch_bwd_wgmma<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                    d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
      else
        launch_bwd_wgmma<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
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
      if (causal)
        lse_mma_kernel_bal<512, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                      Hkv, scale);
      else
        lse_mma_kernel<512><<<lg, THREADS, kLseSmem>>>(d_q, d_k, d_lse, S, H, Hkv, scale,
                                                       (int)causal);
      if (delta_warp_sel)
        delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      else
        delta_kernel<512><<<pg, THREADS>>>(d_o, d_do, d_delta, S, H);
    } else {
#if defined(FA_WGMMA) && defined(FA_TMA)
      if (causal && lse_tma)
        lse_mma_kernel_bal_tma<128, 1><<<lg_bal, THREADS, kLseSmemTma1>>>(qmap_lse, kmap_lse,
                                                                         d_lse, S, H, Hkv,
                                                                         scale);
      else
#endif
      if (causal && lse_wgm)
        lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(d_q, d_k, d_lse,
                                                                            S, H, Hkv, scale);
      else if (causal)
        lse_mma_kernel_bal<128, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                      Hkv, scale);
      else
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
    // O13：HD=128（NDT==1）时 dQ 由主 kernel **覆盖写**（寄存器累加后一次写回），无需清零；
    // 只有 MLA（HD=512）的 GEMM5 走全局 RMW 累加才需要 memset。省掉一趟 n 个 float 的 memset。
    if (D == 512) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
    run_pre();
    run_main();
    // O24：dq_direct 时主 kernel 已直接写 fp16 dq ⇒ convert 跳过 dQ（n_q=0）。
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

  if (D == 512) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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

  // ---- O8b A/B（仅 causal，HD=128）：LSE 原版(O8) vs 镜像配对 vs 镜像配对+cp.async ----
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
    // ---- O30 A/B：LSE 的 wgmma+cp.async 版 vs 4D-TMA 版（同 session + 数值对拍）----
    if (lse_tma) {
      float* d_lse2 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lse2, (size_t)B * S * H * sizeof(float)));
      auto launch_w = [&]() {
        lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(d_q, d_k, d_lse2,
                                                                            S, H, Hkv, scale);
      };
      auto launch_t = [&]() {
        lse_mma_kernel_bal_tma<128, 1><<<lg_bal, THREADS, kLseSmemTma1>>>(qmap_lse, kmap_lse,
                                                                         d_lse, S, H, Hkv,
                                                                         scale);
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
      printf("[O30 A/B] lse tma %.4f ms | wgmma %.4f ms (wgmma/tma %.3fx) | "
             "max_abs(tma-vs-wgmma)=%.3e\n", ms_t, ms_w, ms_w / ms_t, e);
      cudaFree(d_lse2);
    }
#endif
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
    if (D == 512) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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

  // ---- O16 A/B（仅 FA_WGMMA 构建）：wgmma 主 kernel 分段 wait_group（重叠 epilogue）0 vs 1 ----
#ifdef FA_WGMMA
  if (wgmma_sel && D == 128 && bm_sel == 64 && bn_sel == 64) {
    auto time_wgm = [&](bool ow, float* out_ms, float* dq_c, float* dk_c, float* dv_c) {
      dim3 g((S + 63) / 64, H, B);
      auto launch = [&]() {
        if (ow)
          launch_bwd_wgmma<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, 0);
        else
          launch_bwd_wgmma<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, 0);
      };
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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
      CUDA_CHECK(cudaMemcpy(dq_c, d_dq_acc, n * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk_c, d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv_c, d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    };
    std::vector<float> q0(n), k0(nkv), v0(nkv), q1(n), k1(nkv), v1(nkv);
    float ms_ow0 = 0.f, ms_ow1 = 0.f;
    time_wgm(false, &ms_ow0, q0.data(), k0.data(), v0.data());
    time_wgm(true, &ms_ow1, q1.data(), k1.data(), v1.data());
    double dq_d = 0, dk_d = 0, dv_d = 0;
    for (size_t i = 0; i < n; ++i) dq_d = std::max(dq_d, (double)std::fabs(q0[i] - q1[i]));
    for (size_t i = 0; i < nkv; ++i) {
      dk_d = std::max(dk_d, (double)std::fabs(k0[i] - k1[i]));
      dv_d = std::max(dv_d, (double)std::fabs(v0[i] - v1[i]));
    }
    printf("[O16 A/B] main wgmma wait0 %.4f ms | wait_group(重叠) %.4f ms (%.3fx) | "
           "max|diff| dq/dk/dv=%.2e/%.2e/%.2e\n",
           ms_ow0, ms_ow1, ms_ow0 / ms_ow1, dq_d, dk_d, dv_d);
  }
#endif

  // ---- O17 A/B（仅 FA_WGMMA 构建，HD=128）：mma 最优档 vs O9b wgmma vs O17 wgmma2(BM=128) ----
#ifdef FA_WGMMA
  if (D == 128) {
    auto time_o17 = [&](int mode, float* out_ms, float* dq_c, float* dk_c, float* dv_c) {
      auto launch = [&]() {
        if (mode == 4) {
          dim3 g((S + 255) / 256, H, B);
          launch_bwd_wgmma4<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 3) {
          dim3 g((S + 255) / 256, H, B);
          launch_bwd_wgmma4<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 2) {
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
        } else if (mode == 1) {
          dim3 g((S + 63) / 64, H, B);
          launch_bwd_wgmma<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, 0);
        } else {
          launch_cfg(bm_sel, bn_sel, pp_sel, false, true);
        }
      };
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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
      CUDA_CHECK(cudaMemcpy(dq_c, d_dq_acc, n * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk_c, d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv_c, d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    };
    std::vector<float> q0(n), k0(nkv), v0(nkv), q1(n), k1(nkv), v1(nkv), q2(n), k2(nkv),
        v2(nkv), q3(n), k3(nkv), v3(nkv), q4(n), k4(nkv), v4(nkv), q5(n), k5(nkv), v5(nkv);
    std::vector<float> q6(n), k6(nkv), v6(nkv), q7(n), k7(nkv), v7(nkv);
    float ms_mma = 0.f, ms_wgm = 0.f, ms_wg2 = 0.f, ms_wg4 = 0.f, ms_wg4s = 0.f;
    float ms_wg2ns = 0.f, ms_wg2b = 0.f, ms_wg2bns = 0.f;
    time_o17(0, &ms_mma, q0.data(), k0.data(), v0.data());
    time_o17(1, &ms_wgm, q1.data(), k1.data(), v1.data());
    time_o17(2, &ms_wg2, q2.data(), k2.data(), v2.data());
    time_o17(3, &ms_wg4, q3.data(), k3.data(), v3.data());
    time_o17(4, &ms_wg4s, q4.data(), k4.data(), v4.data());
    time_o17(5, &ms_wg2ns, q5.data(), k5.data(), v5.data());
    time_o17(6, &ms_wg2b, q6.data(), k6.data(), v6.data());
    time_o17(7, &ms_wg2bns, q7.data(), k7.data(), v7.data());
    auto mad = [](const std::vector<float>& a, const std::vector<float>& b) {
      double d = 0; for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i]-b[i])); return d;
    };
    printf("[O17 A/B] main mma(%d,%d,%d) %.4f ms | O9b wgmma %.4f (%.3fx) | O17 wgmma2(BM128) "
           "%.4f (%.3fx) TF %.1f\n",
           bm_sel, bn_sel, pp_sel, ms_mma, ms_wgm, ms_mma / ms_wgm, ms_wg2, ms_mma / ms_wg2,
           main_flops / (ms_wg2 * 1e-3) / 1e12);
    printf("[O17b A/B] main O17(BM128) %.4f ms | O17b wgmma4(BM256,ovlp) %.4f ms (%.3fx) "
           "| wgmma4(BM256,seq) %.4f ms (%.3fx) TF %.1f\n",
           ms_wg2, ms_wg4, ms_wg2 / ms_wg4, ms_wg4s, ms_wg2 / ms_wg4s,
           main_flops / (ms_wg4s * 1e-3) / 1e12);
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
    printf("[O17-2 A/B] main O17 wg2 wg0串行(dV+dK) %.4f ms (%.1f TF) | O17-2 拆分(wg0=dV,"
           "wg1=dK) %.4f ms (%.1f TF) => %.3fx | max|diff| dq/dk/dv=%.2e/%.2e/%.2e\n",
           ms_wg2ns, main_flops / (ms_wg2ns * 1e-3) / 1e12, ms_wg2,
           main_flops / (ms_wg2 * 1e-3) / 1e12, ms_wg2ns / ms_wg2, mad(q2, q5), mad(k2, k5),
           mad(v2, v5));
    printf("[O17b A/B] max|diff| wg4-vs-wg2 dq/dk/dv=%.2e/%.2e/%.2e | wg4-vs-mma "
           "dq/dk/dv=%.2e/%.2e/%.2e\n",
           mad(q3, q2), mad(k3, k2), mad(v3, v2), mad(q3, q0), mad(k3, k0), mad(v3, v0));
    printf("[O17b A/B] max|diff| wg4seq-vs-wg4 dq/dk/dv=%.2e/%.2e/%.2e | wg4seq-vs-wg2 "
           "dq/dk/dv=%.2e/%.2e/%.2e\n",
           mad(q4, q3), mad(k4, k3), mad(v4, v3), mad(q4, q2), mad(k4, k2), mad(v4, v2));
  }
#endif

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
  }  // end if (D == 128)

  // ---- O5c A/B（仅 HD=128 且 causal）：不同 (BM,BN,PIPE) tile 配置 ----
  if (D == 128 && causal) {
    auto time_cfg = [&](int bm, int bn, int pp, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, false, true); };
      if (D == 512) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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

  // ---- O7c A/B（仅 HD=128 且 causal）：在 4 个几何上对比 {float2/float4} × {无/有 LSE-D 预装} ----
  if (D == 128 && causal) {
    auto time_r4 = [&](int bm, int bn, int pp, bool r4, bool prel, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, r4, prel); };
      if (D == 512) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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

  // ---- MLA（HD=512）配置 A/B：BM=32/64 × PIPE=0/1，看哪种并行度/smem 组合最快 ----
  if (D == 512 && causal) {
    auto time_cfg2 = [&](int bm, int bn, int pp, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, false, true); };
      if (D == 512) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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

  // ---- O7b A/B（仅 FA_WGMMA 构建、HD=128，`--det` 开启）：跨 CTA `atomicAdd` vs
  //      确定性 partial 缓冲 + 二次归约。跑两遍 DET 验证逐位可复现，再与 atomic 比 max|diff|。
#ifdef FA_WGMMA
  if (det_ab && D == 128) {
    const int nblk = (S + 127) / 128;
    const size_t part_elems = (size_t)B * H * nblk * S * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    dim3 g((S + 127) / 128, H, B);
    auto run_atomic = [&]() {
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      launch_bwd_wgmma2b<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                    d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
    };
    auto run_det = [&]() {
      launch_bwd_wgmma2b<128, true, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                          d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                                          d_dk_part, d_dv_part, nblk);
      dim3 rg(B * Hkv, S);
      dkv_reduce_kernel<128><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H, Hkv,
                                          nblk, (int)causal);
    };
    auto time_fn = [&](auto fn, float* out_ms) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_at = 0.f, ms_det = 0.f;
    time_fn(run_atomic, &ms_at);
    std::vector<float> ak(nkv), av(nkv);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    time_fn(run_det, &ms_det);
    std::vector<float> dk1(nkv), dv1(nkv);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    run_det();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> dk2(nkv), dv2(nkv);
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    auto mad = [](const std::vector<float>& a, const std::vector<float>& b) {
      double d = 0;
      for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i] - b[i]));
      return d;
    };
    printf("[O7b A/B] main wgmma2b atomic %.4f ms (%.1f TF) | DET(partial+reduce) %.4f ms (%.3fx) "
           "| runs[1-2] bitwise-diff dk/dv=%.2e/%.2e | DET-vs-atomic dk/dv=%.2e/%.2e\n",
           ms_at, main_flops / (ms_at * 1e-3) / 1e12, ms_det, ms_at / ms_det, mad(dk1, dk2),
           mad(dv1, dv2), mad(dk1, ak), mad(dv1, av));
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
  }
#endif

  // ---- 数值对拍（重新跑一次完整 forward 保证累加缓冲清零）----
  run_all();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> hdq(n), hdk(nkv), hdv(nkv);
  CUDA_CHECK(cudaMemcpy(hdq.data(), dq, n * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdk.data(), dk, nkv * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdv.data(), dv, nkv * sizeof(__half), cudaMemcpyDeviceToHost));
  std::vector<float> mdq(n), mdk(nkv), mdv(nkv);
  for (size_t i = 0; i < n; ++i) mdq[i] = __half2float(hdq[i]);
  for (size_t i = 0; i < nkv; ++i) {
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
