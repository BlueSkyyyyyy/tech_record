// =============================================================================
// fa_bwd_fp8_main.cu —— FlashAttention 反向（FP8）**两文件版的 host 部分**
// =============================================================================
// 由 P3-4 单文件 `fa_bwd_fp8_mma_onefile.cu` 拆分而来（P3-5）：device 代码移入
// `fa_bwd_fp8_kernels.cuh`，本文件只保留 host 侧：npy 读取 / launcher / 自测对拍。
// 行为与单文件版**逐指标一致**（kernel 代码逐字未改）。
//
// P5-3：按 head_dim 分派模板实例。HD=128 用 (BM=64,BN=32)（MHA/GQA，逐位回归）；
// HD=512 用 (BM=64,BN=32)（MLA 主注意力，smem ~213KB，1 CTA/SM）。
//
// 用法：run.sh src/fp8/fa_bwd_fp8_main.cu [--dir=...] [--full|--causal] [--o=...] [--iters=N]
// =============================================================================

#include "fa_bwd_fp8_kernels.cuh"

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
// O32：为 LSE 的 Q/K 建 4D TMA 描述符（dims={D,S,H,B}，SW128，box={128,64,1,1}）。
// fp8 一行 = 128 字节 = SW128 atom 整行 ⇒ 一个 box 覆盖整个 head_dim（不像 fp16 需 2 chunk）。
// dtype 用 UINT8（CUDA 13 驱动枚举无 FLOAT8_E4M3）。globalStride（字节）：dim1(S) 行距 = H*D，
// dim2(H) 头距 = D，dim3(B) 批距 = S*H*D（元素即字节）。要求 16B 对齐（D=128 恒成立）。
static CUtensorMap make_lse_map_fp8(const void* ptr, long long H, long long S, long long D,
                                    long long B, uint32_t boxR = 64) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D), (uint64_t)D, (uint64_t)(S * H * D)};
  uint32_t box[4] = {128, boxR, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, (void*)ptr, dims, strides, box, estr,
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
    if (d > st.max_abs) st.max_abs = d;
    double r = d / (std::fabs((double)b[i]) + 1e-3);
    if (r > st.max_rel) st.max_rel = r;
  }
  return st;
}

// =============================================================================
// 模板 launcher：按 HD 选择实例并设置动态 smem 上限。
// =============================================================================
// O7：REGDQ 选择是否把 dQ 沿 nt 累加在寄存器里（见 kernels.cuh 主 kernel 说明）。
// O9c-2：WGMMA=true 时 GEMM1/2 走 wgmma（Q/dO/K/V 存 SW128），smem 用 wgmma 布局。
// O12：PREL=true 时把本线程负责的 LSE/D 预装寄存器（见 kernels.cuh），默认开。
// O7e-2：F16B=true 时 fold 的 Ap/dS3/dS2 用 16B 向量化写（见 kernels.cuh），默认开。
// O27：RCP=true 时 fold 量化用「每行 rcp + 乘法」代替逐元素精确除法（见 kernels.cuh）。
template <int HD, int BM, int BN, bool REGDQ, bool WGMMA = false, bool PREL = true, bool F16B = true,
          bool RCP = true>
static void launch_bwd_main(dim3 mg, const unsigned char* q8, const float* qs,
                            const unsigned char* k8, const float* ks,
                            const unsigned char* v8, const float* vs,
                            const unsigned char* do8, const float* dos,
                            const float* delta, const float* lse, float* dq_acc,
                            float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                            float scale, int causal, int ksplit,
                            const int* cu_seqlens = nullptr,
                            const int* mt_b = nullptr, const int* mt_m = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = WGMMA ? Cfg::smem_bytes_wgmma : Cfg::smem_bytes;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP><<<mg, THREADS, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit, cu_seqlens, mt_b, mt_m);
}

// O37：Q/dO 4D-TMA 版主 kernel（仅 `-DFA_WGMMA -DFA_TMA` 构建、HD=128、WGMMA 路径）。
#if defined(FA_WGMMA) && defined(FA_TMA)
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true>
static void launch_bwd_main_qdtma(dim3 mg, const CUtensorMap& qmap, const CUtensorMap& dmap,
                                  const unsigned char* q8, const float* qs,
                                  const unsigned char* k8, const float* ks,
                                  const unsigned char* v8, const float* vs,
                                  const unsigned char* do8, const float* dos,
                                  const float* delta, const float* lse, float* dq_acc,
                                  float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                  float scale, int causal, int ksplit) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_wgmma_tma;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_qdtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_qdtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP>
      <<<mg, THREADS, kSmem>>>(qmap, dmap, q8, qs, k8, ks, v8, vs, do8, dos, delta, lse,
                               dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, ksplit);
}

// O41：Q/dO/K/V 全 4D-TMA 版主 kernel（roadmap「下一步候选 ①」；仅 `-DFA_WGMMA -DFA_TMA`
//   构建、HD=128、WGMMA 路径）。K 双缓冲、V 单缓冲，Kp 由 SW128 K tile 重建。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true>
static void launch_bwd_main_kvtma(dim3 mg, const CUtensorMap& qmap, const CUtensorMap& dmap,
                                  const CUtensorMap& kmap, const CUtensorMap& vmap,
                                  const unsigned char* q8, const float* qs,
                                  const unsigned char* k8, const float* ks,
                                  const unsigned char* v8, const float* vs,
                                  const unsigned char* do8, const float* dos,
                                  const float* delta, const float* lse, float* dq_acc,
                                  float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                  float scale, int causal, int ksplit) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_wgmma_kvtma;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP>
      <<<mg, THREADS, kSmem>>>(qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos,
                               delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal,
                               ksplit);
}
#endif

// O19：跨 warpgroup 归约版主 kernel（BM=128, 2 wg, 256 线程）的 smem 与 launcher。
template <int HD>
static constexpr int wg2_smem_bytes() {
  constexpr int BM = 128, BN = 32;
  constexpr int ASLD = HD + 16, PSLD = HD + 8, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  return (2 * BM * ASLD + 2 * BN * ASLD) + 2 * (BM / 2) * PSLD * 2 + (BN / 2) * PSLD * 2 +
         BM * DSS2 + (kNScale + 2 * BM * PSS) * (int)sizeof(float);
}

template <int HD>
static void launch_bwd_wg2(dim3 mg, const unsigned char* q8, const float* qs,
                           const unsigned char* k8, const float* ks,
                           const unsigned char* v8, const float* vs,
                           const unsigned char* do8, const float* dos,
                           const float* delta, const float* lse, float* dq_acc,
                           float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                           float scale, int causal, int ksplit) {
  constexpr int kSmem = wg2_smem_bytes<HD>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wg2_kernel<HD, 128, 32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wg2_kernel<HD, 128, 32><<<mg, 256, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}

template <int HD>
static void launch_lse(dim3 lg, const unsigned char* q8, const float* qs,
                       const unsigned char* k8, const float* ks, float* lse, int S, int H,
                       int Hkv, float scale, int causal,
                       const int* cu_seqlens = nullptr) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<HD>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  Cfg::lse_smem_bytes));
  lse_mma_kernel<HD><<<lg, THREADS, Cfg::lse_smem_bytes>>>(q8, qs, k8, ks, lse, S, H, Hkv,
                                                           scale, causal, cu_seqlens);
}

// O11：LSE 的镜像配对（+可选 cp.async 双缓冲）版本，仅 causal。
template <int HD, int PIPE>
static void launch_lse_bal(dim3 lg, const unsigned char* q8, const float* qs,
                           const unsigned char* k8, const float* ks, float* lse, int S, int H,
                           int Hkv, float scale, const int* cu = nullptr,
                           float* lse_part = nullptr, int ksplit = 1,
                           long long merge_rows = -1) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = PIPE ? Cfg::lse_smem_bytes_bal1 : Cfg::lse_smem_bytes_bal0;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<HD, PIPE>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  // O39：K 维 split —— grid.z 由 B 扩成 B*ksplit，部分结果写 lse_part 后由 merge kernel 汇总。
  const int B = (int)lg.z;
  dim3 g(lg.x, lg.y, (unsigned)((size_t)B * ksplit));
  lse_mma_kernel_bal<HD, PIPE><<<g, THREADS, kSmem>>>(q8, qs, k8, ks, lse, S, H, Hkv, scale,
                                                       cu, lse_part, ksplit);
  if (ksplit > 1) {
    const long long nrows = (merge_rows >= 0) ? merge_rows : (long long)B * S * H;
    const int th = 256;
    const long long bl = (nrows + th - 1) / th;
    lse_split_merge_kernel<<<(unsigned)bl, th>>>(lse_part, lse, nrows, ksplit);
  }
}

// O9c：fp8 wgmma 版 LSE（SW128 + wgmma.m64n64k32），仅 `-DFA_WGMMA` 构建存在。
#ifdef FA_WGMMA
template <int HD, int PIPE>
static void launch_lse_bal_wgmma(dim3 lg, const unsigned char* q8, const float* qs,
                                 const unsigned char* k8, const float* ks, float* lse, int S,
                                 int H, int Hkv, float scale,
                                 const int* cu_seqlens = nullptr,
                                 const int* pt_b = nullptr, const int* pt_pair = nullptr,
                                 float* lse_part = nullptr, int ksplit = 1,
                                 long long merge_rows = -1) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = PIPE ? Cfg::lse_smem_bytes_balw1 : Cfg::lse_smem_bytes_balw0;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_wgmma<HD, PIPE>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  // O40：K 维 split（对齐 O38/O39）。紧凑网格（pt_pair 非空）时 grid.z 只编 `ksp`（b 由表给出），
  //   否则编 `(b,ksp)`。`ksplit==1` 时内核逐位退化为 O9c 原路径、不写 part、不 launch merge。
  const int B = (int)lg.z;
  dim3 g(lg.x, lg.y, pt_pair ? (unsigned)ksplit : (unsigned)((size_t)B * ksplit));
  lse_mma_kernel_bal_wgmma<HD, PIPE><<<g, THREADS, kSmem>>>(q8, qs, k8, ks, lse, S, H, Hkv,
                                                             scale, cu_seqlens, pt_b, pt_pair,
                                                             lse_part, ksplit);
  if (ksplit > 1) {
    const long long nrows = (merge_rows >= 0) ? merge_rows : (long long)B * S * H;
    const int th = 256;
    const long long bl = (nrows + th - 1) / th;
    lse_split_merge_kernel<<<(unsigned)bl, th>>>(lse_part, lse, nrows, ksplit);
  }
}
#endif

// O32：fp8 TMA 版 LSE（4D-TMA 载入 Q/K，单块），仅 `-DFA_WGMMA -DFA_TMA` 构建存在。
#if defined(FA_WGMMA) && defined(FA_TMA)
template <int HD, int PIPE>
static void launch_lse_bal_tma(dim3 lg, const CUtensorMap& qmap, const CUtensorMap& kmap,
                               const float* qs, const float* ks, float* lse, int S, int H,
                               int Hkv, float scale) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = Cfg::lse_smem_bytes_tma1;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<HD, PIPE>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  lse_mma_kernel_bal_tma<HD, PIPE><<<lg, THREADS, kSmem>>>(qmap, kmap, qs, ks, lse, nullptr,
                                                           S, H, Hkv, scale, 1);
}

// O38：LSE 的 K 维 split + 二次归约（仅 D=128/causal/TMA）。`lg.z` 是 batch B；内部把
//   grid.z 扩成 `B*ksplit`，kernel 每个 (pair,ks) 只扫本 m 块的第 ks 个 K tile 切片，把
//   部分 (m,l) 写入 `lse_part`（[B*S*H][ksplit] 个 (m,l)），随后 merge kernel 汇总成 `lse`。
//   `ksplit==1` 时等于原 O32 路径（kernel 内逐位退化，不写 part、不 launch merge）。
//   好处：小 S / 低 H 时镜像配对后的 grid（=`ceil(nblk/2)*H`）常 < 132 SM，split 直接补满
//   并发槽、缩短「单 CTA 顺序扫 nblk+1 个 tile」的临界路径；大 S 已铺满一个波时收益趋零。
template <int HD, int PIPE>
static void launch_lse_bal_tma_split(dim3 lg, const CUtensorMap& qmap, const CUtensorMap& kmap,
                                     const float* qs, const float* ks, float* lse,
                                     float* lse_part, int S, int H, int Hkv, float scale,
                                     int ksplit) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = Cfg::lse_smem_bytes_tma1;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<HD, PIPE>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const int B = (int)lg.z;
  dim3 g(lg.x, lg.y, (unsigned)(B * ksplit));
  lse_mma_kernel_bal_tma<HD, PIPE><<<g, THREADS, kSmem>>>(qmap, kmap, qs, ks, lse, lse_part,
                                                          S, H, Hkv, scale, ksplit);
  if (ksplit > 1) {
    const long long nrows = (long long)B * S * H;
    const int th = 256;
    const long long bl = (nrows + th - 1) / th;
    lse_split_merge_kernel<<<(unsigned)bl, th>>>(lse_part, lse, nrows, ksplit);
  }
}
#endif

// =============================================================================
// VARLEN（变长 / cu_seqlens）自测入口（fp8，HD=128，causal/full，非 TMA 路径）
// =============================================================================
// 输入是 **packed** 布局 [T,H,D]（q/dO）与 [T,Hkv,D]（k/v），外加 `cu_seqlens.npy`
// （float32 保存，值即 token 前缀和，B+1 个）。kernel 侧用 `cu_seqlens[b]` 作 token 基址、
// `S` 参数传各序列最大长度 maxlen；每个 (b,h,mblk) 只处理自己序列内的 tile。
// ref 输出 `ref_dq/dk/dv.npy` 也是 packed 布局。FA/TE 不支持变长（本机版本），只对 fp32 ref。
// causal / 非 causal 均支持：causal 走镜像配对的 wgmma LSE，非 causal（各块工作量相同）走
// O1 的 mma LSE；非 TMA 路径（LSE 用 wgmma/mma，主 kernel Q/dO 用 cp.async）。
static int run_varlen(const std::string& dir, bool causal, int iters, bool compact = false,
                      bool lse_compact = false, int lse_split = 0) {
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
  const size_t nq = (size_t)T * H * D;       // packed q/dO/dq 长度
  const size_t nkv = (size_t)T * Hkv * D;    // packed k/v/dk/dv 长度
  const size_t rows_q = (size_t)T * H;
  const size_t rows_kv = (size_t)T * Hkv;
  const float scale = 1.0f / sqrtf((float)D);
  printf("case = %s\n", dir.c_str());
  printf("VARLEN: B=%d T=%d maxlen=%d H=%d Hkv=%d D=%d causal=%d\n", B, T, maxlen, H, Hkv, D,
         (int)causal);
  printf("cu_seqlens =");
  for (int b = 0; b <= B && b < 12; ++b) printf(" %d", (int)cu_np.data[b]);
  printf("%s\n", (B > 11) ? " ..." : "");

  float *d_q_f, *d_k_f, *d_v_f, *d_do_f, *d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos, *d_delta, *d_lse, *d_dq, *d_dk, *d_dv;
  float* d_lse_part = nullptr;   // O40：varlen LSE 的 K 维 split 部分结果（[T*H][ksplit] 个 (m,l)）
  int* d_cu;
  CUDA_CHECK(cudaMalloc(&d_q_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_k_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_v_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_do_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_q8, nq));
  CUDA_CHECK(cudaMalloc(&d_k8, nkv));
  CUDA_CHECK(cudaMalloc(&d_v8, nkv));
  CUDA_CHECK(cudaMalloc(&d_do8, nq));
  CUDA_CHECK(cudaMalloc(&d_qs, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_ks, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_vs, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_dos, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_delta, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_lse, rows_q * 4));
  // O40：按最大 split=16 预留（与定长路径一致；实际只用到 auto 选出的 ksplit）。
  CUDA_CHECK(cudaMalloc(&d_lse_part, rows_q * 16 * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_dk, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dv, nkv * 4));
  std::vector<int> cu(B + 1);
  for (int b = 0; b <= B; ++b) cu[b] = (int)cu_np.data[b];
  CUDA_CHECK(cudaMalloc(&d_cu, (B + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_cu, cu.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
  // 均衡分块表：把每个 b 的有效 m 块 (mblk, b) 按 (b,mblk) 主序写进一维表，主 kernel 用
  //   blockIdx.x 查表得到 (b, mblk)（见 fp8_mma_body 顶部注释）。只发有效 tile，消死 CTA。
  constexpr int BV = 64;
  std::vector<std::pair<int,int>> tiles;   // (mblk, b)
  for (int b = 0; b < B; ++b) {
    int nb = (cu[b + 1] - cu[b] + BV - 1) / BV;
    for (int mb = 0; mb < nb; ++mb) tiles.push_back({mb, b});
  }
  // 保持「b 主序、mblk 次序」以复用同序列 K/V 的 L2 局部性（按 mblk 排序会把不同
  // 序列交错，实测反而慢 —— 见文档）。
  const int total_mt = (int)tiles.size();
  std::vector<int> h_mtb(total_mt), h_mtm(total_mt);
  for (int i = 0; i < total_mt; ++i) { h_mtm[i] = tiles[i].first; h_mtb[i] = tiles[i].second; }
  int *d_mtb = nullptr, *d_mtm = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mtb, std::max(1, total_mt) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_mtm, std::max(1, total_mt) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_mtb, h_mtb.data(), total_mt * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mtm, h_mtm.data(), total_mt * sizeof(int), cudaMemcpyHostToDevice));
  // LSE 的镜像对表：每个 b 的有效对 = ceil(nblk_b/2)（nblk_b 按 LBM=64），按工作量降序。
  constexpr int LBMH = 64;
  std::vector<std::pair<int,int>> pairs;   // (pair, b)
  for (int b = 0; b < B; ++b) {
    int nb = (cu[b + 1] - cu[b] + LBMH - 1) / LBMH;
    int np = (nb + 1) / 2;
    for (int p = 0; p < np; ++p) pairs.push_back({p, b});
  }
  std::stable_sort(pairs.begin(), pairs.end(),
                   [&](const std::pair<int,int>& x, const std::pair<int,int>& y) {
                     int nx = (cu[x.second + 1] - cu[x.second] + LBMH - 1) / LBMH;
                     int ny = (cu[y.second + 1] - cu[y.second] + LBMH - 1) / LBMH;
                     return nx > ny;
                   });
  const int total_pairs = (int)pairs.size();
  std::vector<int> h_ptb(total_pairs), h_ptp(total_pairs);
  for (int i = 0; i < total_pairs; ++i) { h_ptp[i] = pairs[i].first; h_ptb[i] = pairs[i].second; }
  int *d_ptb = nullptr, *d_ptp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ptb, std::max(1, total_pairs) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ptp, std::max(1, total_pairs) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ptb, h_ptb.data(), total_pairs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ptp, h_ptp.data(), total_pairs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  // delta = rowsum(dO∘O) 需要前向输出 O（fp32 ref_o，与 dO 同 dtype/布局）。
  CUDA_CHECK(cudaMalloc(&d_o_f, nq * 4));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), nq * 4, cudaMemcpyHostToDevice));

  // ksplit / REGDQ 自动档：与定长路径同公式（O29），用 maxlen 作为串长。
  //   D==128：S>=2048 → 8192，否则 max(2048, 4*base_grid)；
  //   D==512（MLA）：target = len/2（O29 标定），regdq 恒关（HD=512 的 dQ 一次铺不满 N）。
  constexpr int BM = 64;
  const long base_grid = (long)((maxlen + BM - 1) / BM) * H * B;
  const long target_ctas = (D == 128)
                               ? ((maxlen >= 2048) ? 8192L : std::max(2048L, 4L * base_grid))
                               : (long)(maxlen / 2);
  long kk = target_ctas / base_grid; if (kk < 1) kk = 1; if (kk > 16) kk = 16;
  long kp = 1; while (kp * 2 <= kk) kp *= 2;
  const int ksplit = (int)kp;
  const bool use_regdq = (D == 128) && ((long)(maxlen / 32) / 2 / ksplit >= 4);

  // O40：varlen LSE 的 K 维 split auto（D=128 目标 `grid*split≈2048`、cap 8；D=512 `≈256`、
  //   cap 16，与定长 O38/O39 同标定）；`--lsesplit=N`（>0）直接指定。再按最大序列的 tile 数封顶。
  const int nblk0 = (maxlen + LBM - 1) / LBM;
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    long base = (D == 128 && lse_compact) ? (long)total_pairs * H
                                          : (long)((nblk0 + 1) / 2) * H * B;
    const int target = (D == 512) ? 256 : 2048;
    const int cap = (D == 512) ? 16 : 8;
    int sp = 1;
    while (sp < cap && base * (sp * 2) <= target) sp *= 2;
    while (sp > nblk0 && sp > 1) sp >>= 1;
    lse_split_eff = sp;
  }
  printf("O40: varlen lse k-split = %d (base=%ld)\n", lse_split_eff,
         (D == 128 && lse_compact) ? (long)total_pairs * H : (long)((nblk0 + 1) / 2) * H * B);

  auto run_all = [&]() {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const int gq = (int)std::min<long long>((rq + 3) / 4, 65535);
    const int gkv = (int)std::min<long long>((rkv + 3) / 4, 65535);
    // VPT = D/32：D=128→4、D=512→16（与定长路径一致；写错会把行距当 128 ⇒ 整行错位）。
    if (D == 128) {
      quantize_row_warp_kernel<4, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<4, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<4, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<4, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    } else {
      quantize_row_warp_kernel<16, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<16, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    }
    CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
    // LSE：grid.x 按 maxlen，逐 b 由 cu_seqlens 定界。
    //   causal → 镜像配对 + cp.async 的 wgmma 版（工作量随 mblk 递增，需均衡）；
    //   非 causal → 各 m 块工作量恒为 nblk 个 tile，本已均衡，走 O1 的 mma `lse_mma_kernel`。
    const int nblk = (maxlen + LBM - 1) / LBM;
    if (causal) {
      dim3 lg((nblk + 1) / 2, H, B);
      // D==128 走 wgmma 版（SW128 + wgmma）；D==512（MLA）只有 mma 版（HD>128 无 SW128 快路）。
      // O42：纯 `sm_90`（无 `-DFA_WGMMA`）构建下没有 wgmma 版 ⇒ D==128 退回 mma 镜像配对版
      //   （此前无条件调用导致 fp8 main 的 `-arch=sm_90` 构建失败，fp16/bf16 无此问题）。
#ifdef FA_WGMMA
      if (D == 128 && lse_compact)
        launch_lse_bal_wgmma<128, 1>(dim3(total_pairs, H, 1), d_q8, d_qs, d_k8, d_ks, d_lse,
                                     maxlen, H, Hkv, scale, d_cu, d_ptb, d_ptp,
                                     d_lse_part, lse_split_eff, (long long)rows_q);
      else if (D == 128)
        launch_lse_bal_wgmma<128, 1>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale,
                                     d_cu, nullptr, nullptr, d_lse_part, lse_split_eff,
                                     (long long)rows_q);
      else
#endif
      if (D == 128)
        launch_lse_bal<128, 1>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, d_cu,
                               d_lse_part, lse_split_eff, (long long)rows_q);
      else
        launch_lse_bal<512, 1>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, d_cu,
                               d_lse_part, lse_split_eff, (long long)rows_q);
    } else {
      dim3 lg(nblk, H, B);
      if (D == 128)
        launch_lse<128>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, 0, d_cu);
      else
        launch_lse<512>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, 0, d_cu);
    }
    const int d_rows = (int)rows_q;
    const int d_wpb = THREADS / 32;
    const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
    if (D == 128)
      delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    else
      delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    // 均衡分块：`--compact` 走紧凑一维 m-tile 网格（grid.z=1，由 mt_b/mt_m 查表解出 (b, mblk)）；
    //   默认走旧的 maxlen 网格（含早退死 CTA，实测反而更快，见 docs/03 §40）。
    dim3 mg = compact ? dim3(total_mt * ksplit, H, 1)
                      : dim3((maxlen + BM - 1) / BM * ksplit, H, B);
    const int* mtb = compact ? d_mtb : nullptr;
    const int* mtm = compact ? d_mtm : nullptr;
    if (D == 512) {
      // MLA：非 wgmma / 非 regdq（与定长 D=512 路径一致）。
      launch_bwd_main<512, 64, 32, false, false, true, true>(
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
          d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
    } else if (use_regdq)
      launch_bwd_main<128, 64, 32, true, true, true, true, true>(
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
          d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
    else
      launch_bwd_main<128, 64, 32, false, true, true, true, true>(
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
          d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
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
    flops += 4.0 * H * L * L * D;  // bwd ≈ 2×fwd，因果只算一半再用系数 2 近似
  }
  printf("[timing] VARLEN total %.4f ms  %.2f TFLOPS (sum_b 4HL^2D)\n", ms,
         flops / (ms * 1e-3) / 1e12);
  if (compact)
    printf("grid main = %d x %d x %d (compact, ksplit=%d, total_mt=%d, old-grid=%d) | use_regdq=%d | T=%d\n",
           total_mt * ksplit, H, 1, ksplit, total_mt, (maxlen + BM - 1) / BM * B, (int)use_regdq, T);
  else
    printf("grid main = %d x %d x %d (nocompact, ksplit=%d, total_mt=%d) | use_regdq=%d | T=%d\n",
           (maxlen + BM - 1) / BM * ksplit, H, B, ksplit, total_mt, (int)use_regdq, T);

  std::vector<float> h_dq(nq), h_dk(nkv), h_dv(nkv);
  CUDA_CHECK(cudaMemcpy(h_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
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

// =============================================================================
// host / launcher / self-test
// =============================================================================
int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8";
  std::string o_name = "ref_o";
  bool causal = true;
  int iters = 20;
  int ksplit = -1;  // -1 = 自动
  int ksplit2_opt = -1;  // O19：wg2 的 ksplit（-1 自动）
  // O22：在 `-DFA_WGMMA`（sm_90a）构建下，默认启用 Hopper 路径（LSE + 主 kernel GEMM1/2 的
  //   wgmma）；sm_90 构建下这两个宏路径不存在，保持 mma。`--lsewgm=0/--wgmma=0` 可显式退回 mma
  //   做 A/B（S=4096 端到端 wgmma 比 mma 快 ~8%：2.70→2.49ms，preprocess 0.40→0.32、main 2.19→2.04）。
#ifdef FA_WGMMA
  int lsewgm = 1;
  int wgmma = 1;
#else
  int lsewgm = 0;
  int wgmma = 0;
#endif
  int wg2 = 0;      // O19：1 = 主 kernel 走跨 warpgroup 归约版（BM=128, 2 wg, 256 线程）
  int prel_opt = -1;  // O12：-1 自动（开）；0/1 强制 LSE/D 预装寄存器开关
  int qfast = 1;      // O14：1 = warp-per-row 向量化量化，0 = 旧 per-row 标量量化（A/B）
  int delta_warp_opt = 1;  // O26：1 = warp-per-row 向量化 delta（默认），0 = 旧 per-row smem 归约（A/B）
  int f16b_opt = 1;   // O7e-2：1 = fold 16B 向量化写（默认），0 = 退回 O7e 的 4B 写（A/B）
  int bn64_opt = 0;   // O21：1 = 主 kernel KV tile BN=64（mma 路径，D=128）
  int cvt_on = 0;     // O21b：1 = 保留冗余的 fp32→fp32 convert 拷贝（默认 0：直接累加进输出）
  int foldrcp_opt = 1;  // O27：1 = fold 量化用「每行 rcp + 乘法」（默认），0 = 精确除法（A/B）
  int regdq_opt = -1; // O22：-1 自动；0/1 强制关/开寄存器 dQ 累加（同 session A/B）
  // O32：LSE 是否用 TMA 版（仅 FA_TMA 构建、D==128、causal）。-1=自动（默认开），0/1 由
  //   `--lsetma=` 强制。
  int lse_tma = -1;
  // O37：主 kernel 的 Q/dO 是否用 4D-TMA（仅 FA_WGMMA+FA_TMA、D==128）。-1=自动（默认关，
  //   作 opt-in 与 cp.async 版同 binary A/B），0/1 由 `--qdtma=` 强制。
  int qd_tma = -1;
  // O41：主 kernel 的 K/V 是否也用 4D-TMA（roadmap「下一步候选 ①」）。仅 `-DFA_WGMMA -DFA_TMA`
  //   构建、D==128；-1=自动（默认开，对齐 O37），0/1 由 `--kvtma=` 强制。
  int kv_tma = -1;
  // O38：LSE 的 K 维 split 数（仅 D=128/causal/TMA 生效）。0=自动（目标 grid*split≈2048、上限 8），
  //   >=1 直接指定（`--lsesplit=N` 走「切片 partial + merge」；`--lsesplit=1` 退回 O32、保持历史逐位）。
  int lse_split = 0;
  int varlen = 0;   // VARLEN：1 = packed [T,H,D] + cu_seqlens.npy（fp8/HD=128/causal）
  int compact_opt = 0;  // 第八十二轮：1 = varlen 主 kernel 紧凑均衡网格（opt-in；实测中性偏负）
  int lse_compact_opt = 0;  // 第八十二轮：1 = varlen causal LSE 紧凑对网格（opt-in，A/B）
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--full") causal = false;
    else if (a == "--varlen") varlen = 1;
    else if (a == "--compact") compact_opt = 1;
    else if (a == "--lsecompact") lse_compact_opt = 1;
    else if (a.rfind("--varlen=", 0) == 0) varlen = atoi(a.c_str() + 9);
    else if (a == "--causal") causal = true;
    else if (a == "--lsewgm") lsewgm = 1;
    else if (a == "--wgmma") wgmma = 1;
    else if (a == "--lsetma") lse_tma = 1;
    else if (a.rfind("--lsewgm=", 0) == 0) lsewgm = atoi(a.c_str() + 9);
    else if (a.rfind("--wgmma=", 0) == 0) wgmma = atoi(a.c_str() + 8);
    else if (a.rfind("--lsetma=", 0) == 0) lse_tma = atoi(a.c_str() + 9);
    else if (a.rfind("--lsesplit=", 0) == 0) lse_split = atoi(a.c_str() + 11);
    else if (a.rfind("--qdtma=", 0) == 0) qd_tma = atoi(a.c_str() + 8);
    else if (a.rfind("--kvtma=", 0) == 0) kv_tma = atoi(a.c_str() + 8);
    else if (a == "--kvtma") kv_tma = 1;
    else if (a == "--wg2") wg2 = 1;
    else if (a == "--bn64") bn64_opt = 1;
    else if (a.rfind("--cvt=", 0) == 0) cvt_on = atoi(a.c_str() + 6);
    else if (a.rfind("--foldrcp=", 0) == 0) foldrcp_opt = atoi(a.c_str() + 10);
    else if (a.rfind("--qfast=", 0) == 0) qfast = atoi(a.c_str() + 8);
    else if (a.rfind("--deltawarp=", 0) == 0) delta_warp_opt = atoi(a.c_str() + 12);
    else if (a.rfind("--regdq=", 0) == 0) regdq_opt = atoi(a.c_str() + 8);
    else if (a.rfind("--prel=", 0) == 0) prel_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--f16b=", 0) == 0) f16b_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--ksplit=", 0) == 0) ksplit = atoi(a.c_str() + 9);
    else if (a.rfind("--ksplit2=", 0) == 0) ksplit2_opt = atoi(a.c_str() + 10);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  if (varlen) return run_varlen(dir, causal, iters, compact_opt, lse_compact_opt, lse_split);

  auto q_np = load_npy_f32(dir + "/q.npy");
  auto k_np = load_npy_f32(dir + "/k.npy");
  auto v_np = load_npy_f32(dir + "/v.npy");
  auto do_np = load_npy_f32(dir + "/do.npy");
  auto o_np = load_npy_f32(dir + "/" + o_name + ".npy");
  auto rdq = load_npy_f32(dir + "/ref_dq.npy");
  auto rdk = load_npy_f32(dir + "/ref_dk.npy");
  auto rdv = load_npy_f32(dir + "/ref_dv.npy");

  if (q_np.shape.size() != 4) {
    fprintf(stderr, "期望 q 为 4D [B,S,H,D]\n");
    return 1;
  }
  if (k_np.shape.size() != 4 || v_np.shape.size() != 4) {
    fprintf(stderr, "期望 k/v 为 4D [B,S,Hkv,D]\n");
    return 1;
  }
  const int B = (int)q_np.shape[0], S = (int)q_np.shape[1];
  const int H = (int)q_np.shape[2], D = (int)q_np.shape[3];
  const int Hkv = (int)k_np.shape[2];   // P5-3：GQA/MQA 的 KV 头数（MHA 时 Hkv==H）
  if (D != 128 && D != 512) {
    fprintf(stderr, "本版本支持 head_dim=128（MHA/GQA）或 512（MLA）；当前 %d\n", D);
    return 1;
  }
  if ((int)v_np.shape[2] != Hkv || (int)v_np.shape[3] != D) {
    fprintf(stderr, "k/v 形状不一致（不支持 Dv!=D）\n");
    return 1;
  }
  if (H % Hkv != 0) {
    fprintf(stderr, "H(%d) 必须是 Hkv(%d) 的整数倍\n", H, Hkv);
    return 1;
  }
  const size_t nq = (size_t)B * S * H * D;      // q/dO/dq 长度
  const size_t nkv = (size_t)B * S * Hkv * D;   // k/v/dk/dv 长度
  const size_t rows_q = (size_t)B * S * H;
  const size_t rows_kv = (size_t)B * S * Hkv;
  const float scale = 1.0f / sqrtf((float)D);

  // 主 kernel tile 固定 BM=64,BN=32；smem 随 HD 变化。
  const int smem_bytes = (D == 128) ? Fp8Cfg<128, 64, 32>::smem_bytes
                                    : Fp8Cfg<512, 64, 32>::smem_bytes;
  const int lse_smem = (D == 128) ? Fp8Cfg<128, 64, 32>::lse_smem_bytes
                                  : Fp8Cfg<512, 64, 32>::lse_smem_bytes;

  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d Hkv=%d D=%d causal=%d scale=%.6f\n", B, S, H, Hkv, D, (int)causal,
         scale);
  printf("FP8 mma: Q/K/V=E4M3, dO=E5M2, dS2/dS3=E5M2, Ap=E4M3 (rowwise); P/dS fp32\n");
  printf("smem = %d bytes (%.1f KB); lse smem = %d bytes (%.1f KB)\n", smem_bytes,
         smem_bytes / 1024.0, lse_smem, lse_smem / 1024.0);
  if (D == 128)
    printf("main wgmma smem = %d bytes (%.1f KB)\n",
           Fp8Cfg<128, 64, 32>::smem_bytes_wgmma, Fp8Cfg<128, 64, 32>::smem_bytes_wgmma / 1024.0);

  float *d_q_f, *d_k_f, *d_v_f, *d_do_f, *d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc, *d_dq, *d_dk, *d_dv;
  CUDA_CHECK(cudaMalloc(&d_q_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_k_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_v_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_do_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_o_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_q8, nq));
  CUDA_CHECK(cudaMalloc(&d_k8, nkv));
  CUDA_CHECK(cudaMalloc(&d_v8, nkv));
  CUDA_CHECK(cudaMalloc(&d_do8, nq));
  CUDA_CHECK(cudaMalloc(&d_qs, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_ks, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_vs, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_dos, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_delta, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_lse, rows_q * 4));
  // O38：LSE split 的部分结果缓冲（[rows_q][ksplit] 个 (m,l)）。按最大 split=16 预留。
  float* d_lse_part = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lse_part, rows_q * 16 * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dq, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_dk, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dv, nkv * 4));
  // O21b：输出本就是 fp32，与 fp32 累加缓冲同 dtype ⇒ 让 main 的 atomicAdd 直接累加到输出，
  //   消掉 `convert_kernel` 这一趟纯 fp32→fp32 拷贝（S=4096 约 0.145ms / 端到端 ~5%）。
  //   把 acc 指针别名到输出即可（所有自测 A/B 仍读写同一份，结果不变）。
  CUDA_CHECK(cudaFree(d_dq_acc)); d_dq_acc = d_dq;
  CUDA_CHECK(cudaFree(d_dk_acc)); d_dk_acc = d_dk;
  CUDA_CHECK(cudaFree(d_dv_acc)); d_dv_acc = d_dv;

  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), nq * 4, cudaMemcpyHostToDevice));

  // O14：量化输入（q/k/v=E4M3、dO=E5M2，rowwise）。`qfast` 选 warp-per-row 向量化版。
  auto quant_old = [&]() {
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_q_f, d_q8, d_qs, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_k_f, d_k8, d_ks, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_v_f, d_v8, d_vs, D, 0);
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_do_f, d_do8, d_dos, D, 1);
  };
  auto quant_new = [&]() {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const int gq = (int)std::min<long long>((rq + 3) / 4, 65535);
    const int gkv = (int)std::min<long long>((rkv + 3) / 4, 65535);
    if (D == 128) {
      quantize_row_warp_kernel<4, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<4, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<4, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<4, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    } else {
      quantize_row_warp_kernel<16, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<16, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    }
  };
  auto quant = [&]() { if (qfast) quant_new(); else quant_old(); };

  // ---- O2b：自动选择 N 方向切块数 ksplit。base = 未切块时的 CTA 数；切块把小 S 时
  //      不足一个波、或大 S 的尾波（partial wave）用更细的 CTA 补满并发槽。
  //      实测（docs/03 §15 的 ksplit sweep）：d128（smem 73.8KB→3 CTA/SM）在
  //      grid≈4096（≈10 个波）时最优；MLA d512（smem 223KB→1 CTA/SM）在 grid≈一个波
  //      （132）时最优——再切只增 prologue 与 dQ atomic 竞争。故按 head_dim 取目标：
  //        TARGET = (D==128) ? 4096 : 132;  k = clamp(TARGET/base, 1, 16) 后向下取 2 的幂。----
  constexpr int BM = 64;
  const long base_grid = (long)((S + BM - 1) / BM) * H * B;
  if (ksplit < 1) {
    // O29：自动切块数重新标定。原公式 `target=(D==128)?4096:132` 是早期（O2b）在
    //   「d128=3 CTA/SM、MLA=1 CTA/SM」下测的，随后的 O3/O4b/O9c-2/O22 等把数据通路改过之后
    //   已明显次优：
    //   * d128 固定 4096 在 base 小（S=1024，base=512）时**过切**——每个 CTA 只有 ~2 个 tile，
    //     且此时 `use_regdq=false`、dQ 逐 tile 跨 CTA red，多切反而慢。实测 S1024H32 auto k=8
    //     0.340 ms vs k=4 **0.302 ms（1.13×）**；GQA kv4 0.329→0.274（1.20×）。
    //   * S=4096 时 4096 又**欠切**（k=4 1.758 vs k=8 1.737，1.01×）。
    //   * MLA（D=512，1 CTA/SM）固定 132（≈1 个波）**欠切**——S1024H2 auto k=4 0.265 ms vs
    //     k=16 **0.201 ms（1.32×）**；S512H4 k=4 0.140 vs k=8 0.122。
    //   新标定（sweep 见 src/fp8/fa_bwd_fp8_o29_ksplit_sweep.out.txt）：
    //     D==128: S>=2048 → 8192；否则 max(2048, 4*base_grid)（覆盖 base∈{128,512,640,1024}）
    //     D==512: S/2（S256H2→128 / S512H4→256 / S1024H2→512，三者都取到各自最优）
    const long target_ctas = (D == 128)
                                 ? ((S >= 2048) ? 8192L : std::max(2048L, 4L * base_grid))
                                 : (long)(S / 2);
    long k = target_ctas / base_grid;
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    long kp = 1;
    while (kp * 2 <= k) kp *= 2;  // 向下取 2 的幂，让 grid 对齐到整数个波附近
    ksplit = (int)kp;
  }
  // O7：只有 HD=128（dQ 一次铺满 N）且「平均每 CTA 的 nt tile 足够多」时才启用寄存器累加。
  // 因果下每 mblk 的 nt tile 数 ≈ (m0+BM)/BN，三角求和 /(mblk·ksplit) 后平均每 CTA
  // ≈ (S/BN)/2/ksplit；阈值取 4（实测 S=1024H32 平均=2、启用反而持平/略慢，S=4096=16 明显收益）。
  bool use_regdq = (D == 128) && ((long)(S / 32) / 2 / ksplit >= 4);
  // O22：`--regdq=0/1` 强制开关（仅同 session A/B 用）；-1 = 用上面的启发式。
  if (regdq_opt >= 0) use_regdq = (D == 128) && (regdq_opt != 0);
  dim3 pg(S, H, B);
  // O26：delta 的 warp-per-row 版 grid-stride 覆盖 B*S*H 行（每 warp 一行）。
  const bool delta_warp_sel = (delta_warp_opt != 0);
  const int d_rows = (int)((size_t)B * S * H);
  const int d_wpb = THREADS / 32;
  const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
  dim3 lg((S + LBM - 1) / LBM, H, B);
  dim3 lg_bal((((S + LBM - 1) / LBM) + 1) / 2, H, B);   // O11：镜像配对，grid.x 减半
  dim3 mg((S + BM - 1) / BM * ksplit, H, B);
  // O19：wg2（BM=128）的自动 ksplit 与 grid（base 减半）。同样以 grid≈4096（3 CTA/SM
  //   的 d128 目标）为准；1 CTA/SM 时用更细的 grid 不利，故对 wg2 用 target=2048。
  const long base_grid2 = (long)((S + 127) / 128) * H * B;
  int ksplit2 = ksplit;
  if (ksplit2 < 1) ksplit2 = 1;
  if (ksplit2_opt >= 1) ksplit2 = ksplit2_opt;
  else {
    const long target_ctas = 2048L;
    long k = target_ctas / (base_grid2 > 0 ? base_grid2 : 1);
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    long kp = 1;
    while (kp * 2 <= k) kp *= 2;
    ksplit2 = (int)kp;
  }
  dim3 mg2((S + 127) / 128 * ksplit2, H, B);
  printf("grid main = %d x %d x %d  (ksplit=%d, base_grid=%ld)\n", mg.x, mg.y, mg.z,
         ksplit, base_grid);
  printf("O19: wg2 grid = %d x %d x %d (ksplit2=%d, base_grid2=%ld)\n", mg2.x, mg2.y, mg2.z,
         ksplit2, base_grid2);
  printf("O7: use_regdq=%d (register dQ accumulation)\n", (int)use_regdq);
  const int cvt_threads = 256;
  const int cvt_blocks = (int)std::min<size_t>((nq + cvt_threads - 1) / cvt_threads, 65535);

  // O32：TMA 版 LSE 需驱动 API（`cuTensorMapEncodeTiled`）⇒ 只有 `-DFA_TMA -lcuda` 构建才编译
  //   该路径；此时 D==128/causal 默认开（对齐 fp16 O30，省 load 指令/地址运算）。
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (D == 128) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<128, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Fp8Cfg<128, 64, 32>::lse_smem_bytes_tma1));
  }
  if (lse_tma < 0) lse_tma = (D == 128 && causal) ? 1 : 0;
#else
  if (lse_tma < 0) lse_tma = 0;
#endif
  printf("O32: lse backend = %s\n", lse_tma ? "tma" : "wgmma/mma");
  // O38：自动 split 档（0=auto）。目标 `grid*split ≈ 2048`（≈2 个满波；实测该目标在各 shape
  //   上距 per-shape 最优 ≤1.2%），上限 8；grid 已够大则退回 1（逐位）。
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    long lg_grid = (long)lg_bal.x * H * B;
    // O39：D=512（MLA，mma LSE）目标 `grid*split ≈ 256`、上限 16；D=128 的 TMA LSE 维持
    //   O38 的 `≈2048`、上限 8。
    const int target = (D == 512) ? 256 : 2048;
    const int cap = (D == 512) ? 16 : 8;
    int sp = 1;
    while (sp < cap && lg_grid * (sp * 2) <= target) sp *= 2;
    // 再按 `nblk=ceil(S/64)` 封顶：切得比 K tile 数还细只会产生空切片 + 每 CTA 的 Q 载入开销。
    int nblk_cap = (S + 63) / 64;
    while (sp > nblk_cap) sp >>= 1;
    lse_split_eff = sp;
  }
#if defined(FA_WGMMA) && defined(FA_TMA)
  // O32：建 LSE 的 Q/K 4D TMA 描述符（一次，供所有 (h,b) CTA 用坐标选择）。
  CUtensorMap qmap_lse, kmap_lse;
  // 注：O32/O9c A/B 段无论 `--lsetma` 取值都会跑 TMA 版 LSE，故 D==128 时始终建描述符
  //     （否则 `--lsetma=0` 会拿未初始化 map 启动 TMA kernel → illegal instruction）。
  if (D == 128) {
    qmap_lse = make_lse_map_fp8(d_q8, H, S, D, B);
    kmap_lse = make_lse_map_fp8(d_k8, Hkv, S, D, B);
  }
  // O37：主 kernel 的 Q/dO TMA 描述符（box={128,BM=64}，与 LSE 同一 dims={D,S,H,B}）。
  if (qd_tma < 0) qd_tma = 1;   // 默认开（对齐 O32 的 lsetma；`--qdtma=0` 供 A/B）
  // O41：K/V TMA 默认开（依赖 Q/dO TMA；`--kvtma=0` 供 A/B）。
  if (kv_tma < 0) kv_tma = 1;
  CUtensorMap qmap_main, dmap_main;
  // 注：O37 A/B 段无论 `--qdtma` 取值都会跑 TMA 版，故 D==128 时始终建描述符
  //     （否则 `--qdtma=0` 会拿未初始化 map 启动 TMA kernel → illegal instruction）。
  if (D == 128) {
    qmap_main = make_lse_map_fp8(d_q8, H, S, D, B);
    dmap_main = make_lse_map_fp8(d_do8, H, S, D, B);
  }
  // O41：主 kernel 的 K/V TMA 描述符（dims={D,S,Hkv,B}，box={128,BN=32}）。
  CUtensorMap kmap_main, vmap_main;
  if (D == 128) {
    kmap_main = make_lse_map_fp8(d_k8, Hkv, S, D, B, 32);
    vmap_main = make_lse_map_fp8(d_v8, Hkv, S, D, B, 32);
  }
#else
  if (qd_tma < 0) qd_tma = 0;
  kv_tma = 0;
#endif
  printf("O37: main qd-tma = %s\n", qd_tma ? "on" : "off");
  printf("O41: main kv-tma = %s\n", kv_tma ? "on" : "off");  printf("O38: lse k-split = auto(%d)\n", lse_split_eff);

  auto run_preprocess = [&]() {
    if (D == 128) {
      // O11：causal 走镜像配对 + cp.async 双缓冲（非 causal 各块工作量相同，走 O1 原版）。
      // O9c：`--lsewgm` 且以 `-DFA_WGMMA` 构建时，causal 走 wgmma.m64n64k32（SW128）。
      if (causal) {
#if defined(FA_WGMMA) && defined(FA_TMA)
        if (lse_tma) {
          // O38：split>1 走「K 维切片 + merge」，=1 逐位退回 O32。
          if (lse_split_eff > 1)
            launch_lse_bal_tma_split<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                             d_lse_part, S, H, Hkv, scale, lse_split_eff);
          else
            launch_lse_bal_tma<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H,
                                       Hkv, scale);
        } else
#endif
#ifdef FA_WGMMA
        if (lsewgm)
          launch_lse_bal_wgmma<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv,
                                       scale, nullptr, nullptr, nullptr, d_lse_part,
                                       lse_split_eff);
        else
#endif
          launch_lse_bal<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                 nullptr, d_lse_part, lse_split_eff);
      } else
        launch_lse<128>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, (int)causal);
      if (delta_warp_sel)
        delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
      else
        delta_kernel<128><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    } else {
      // O39：D=512（MLA）causal LSE 也用 K 维 split（此前只有 D=128/TMA 有）。
      if (causal) {
        if (lse_split_eff > 1)
          launch_lse_bal<512, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                 nullptr, d_lse_part, lse_split_eff);
        else
          launch_lse_bal<512, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
      } else
        launch_lse<512>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, (int)causal);
      if (delta_warp_sel)
        delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
      else
        delta_kernel<512><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    }
  };

  // O12：LSE/D 预装寄存器（默认开），`--prel=0` 关；为同 session A/B 派发到两个模板实例。
  const bool prel_sel = (prel_opt < 0) ? true : (prel_opt != 0);
  // O7e-2：fold 16B 向量化写（默认开），`--f16b=0` 退回 O7e 的 4B 写（仅作 A/B）。
  const bool f16b_sel = (f16b_opt != 0);
  // O27：第 5 个开关 rcp 选 fold 量化用乘法（true，默认）还是精确除法（false，A/B）。
  auto launch128 = [&](bool reg, bool wg, bool prel, bool f16, bool rcp = true) {
#define GO2(REG_, WG_, PREL_, F16_)                                                          \
    if (rcp) {                                                                               \
      launch_bwd_main<128, 64, 32, REG_, WG_, PREL_, F16_, true>(                            \
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,    \
          d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);                         \
    } else {                                                                                 \
      launch_bwd_main<128, 64, 32, REG_, WG_, PREL_, F16_, false>(                           \
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,    \
          d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);                         \
    }
#define GO1(REG_, WG_, PREL_)                                                                \
    if (f16) { GO2(REG_, WG_, PREL_, true); } else { GO2(REG_, WG_, PREL_, false); }
#define GO0(REG_, WG_)                                                                       \
    if (prel) { GO1(REG_, WG_, true); } else { GO1(REG_, WG_, false); }
    if (reg) { if (wg) { GO0(true, true); } else { GO0(true, false); } }
    else     { if (wg) { GO0(false, true); } else { GO0(false, false); } }
#undef GO0
#undef GO1
#undef GO2
  };
  auto run_main = [&]() {
    if (D == 128 && wg2) {
      launch_bwd_wg2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                          d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                          ksplit2);
      return;
    }
    if (D == 128 && bn64_opt) {
      // O21：KV tile BN=64（mma 路径），其余与 BN=32 版逐字同构。grid 不变（BM 仍 64）。
      if (use_regdq)
        launch_bwd_main<128, 64, 64, true, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      else
        launch_bwd_main<128, 64, 64, false, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      return;
    }
    const bool rcp_sel = (foldrcp_opt != 0);
#if defined(FA_WGMMA) && defined(FA_TMA)
    // O41：Q/dO/K/V 全 TMA 版（K 双缓冲、V 单缓冲）。仅默认 fold 选项下启用。
    if (D == 128 && wgmma && qd_tma && kv_tma && prel_sel && f16b_sel && rcp_sel) {
      if (use_regdq)
        launch_bwd_main_kvtma<128, 64, 32, true>(
            mg, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
            d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
            scale, (int)causal, ksplit);
      else
        launch_bwd_main_kvtma<128, 64, 32, false>(
            mg, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
            d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
            scale, (int)causal, ksplit);
      return;
    }
    // O37：Q/dO TMA 版（仅在默认 fold 选项下启用；其它组合回退 cp.async 版）。
    if (D == 128 && wgmma && qd_tma && prel_sel && f16b_sel && rcp_sel) {
      if (use_regdq)
        launch_bwd_main_qdtma<128, 64, 32, true>(
            mg, qmap_main, dmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
            d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
            ksplit);
      else
        launch_bwd_main_qdtma<128, 64, 32, false>(
            mg, qmap_main, dmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
            d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
            ksplit);
      return;
    }
#endif
#ifdef FA_WGMMA
    if (D == 128 && wgmma) { launch128(use_regdq, true, prel_sel, f16b_sel, rcp_sel); return; }
#endif
    if (D == 128) { launch128(use_regdq, false, prel_sel, f16b_sel, rcp_sel); return; }
    if (prel_sel)
      launch_bwd_main<512, 64, 32, false, false, true, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                       d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                       (int)causal, ksplit);
    else
      launch_bwd_main<512, 64, 32, false, false, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                        d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                        (int)causal, ksplit);
  };

  auto run_all = [&]() {
    quant();
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    run_preprocess();
    run_main();
    if (cvt_on)
      convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                  d_dv, nq, nkv);
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
  printf("[timing] total(quant+pre+main+cvt) %.4f ms  %.2f TFLOPS (bwd FLOPs=4BS^2HD)\n", ms,
         flops / (ms * 1e-3) / 1e12);

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) quant();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_quant = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_quant, ev0, ev1));
  ms_quant /= iters;

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_preprocess();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_pre = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev0, ev1));
  ms_pre /= iters;

  CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
  CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_main();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_main = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_main, ev0, ev1));
  ms_main /= iters;
  printf("[timing] quant %.4f ms | preprocess %.4f ms | main %.4f ms | convert %.4f ms (cvt_on=%d)\n",
         ms_quant, ms_pre, ms_main, ms - ms_quant - ms_pre - ms_main, cvt_on);

#ifdef FA_WGMMA
  // ---- O9c-2 A/B（D=128）：主 kernel GEMM1/2 的 mma.m16n8k32 vs wgmma.m64n32k32。----
  //      同一 session 计时 + 逐元素对拍（证明只换计算后端、数学口径未变）。
  if (D == 128) {
    auto run_main_wg = [&](bool wg) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (use_regdq) {
        if (wg)
          launch_bwd_main<128, 64, 32, true, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                   d_vs, d_do8, d_dos, d_delta, d_lse,
                                                   d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                   Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, true, false>(mg, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                    d_vs, d_do8, d_dos, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                    Hkv, scale, (int)causal, ksplit);
      } else {
        if (wg)
          launch_bwd_main<128, 64, 32, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                    d_vs, d_do8, d_dos, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                    Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false>(mg, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                     d_vs, d_do8, d_dos, d_delta, d_lse,
                                                     d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                     Hkv, scale, (int)causal, ksplit);
      }
    };
    auto bench_main_wg = [&](bool wg, float* out) {
      run_main_wg(wg);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_main_wg(wg);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float mmma = 0.f, mwgm = 0.f;
    bench_main_wg(false, &mmma);
    bench_main_wg(true, &mwgm);
    std::vector<float> a_dq(nq), a_dk(nkv), a_dv(nkv), b_dq(nq), b_dk(nkv), b_dv(nkv);
    run_main_wg(false);
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_main_wg(true);
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O9c-2 A/B] main mma(regdq=%d) %.4f ms | wgmma(m64n32) %.4f ms (%.3fx) | "
           "max_abs(wg-vs-mma) dq/dk/dv=%.3e/%.3e/%.3e\n",
           (int)use_regdq, mmma, mwgm, mmma / mwgm, maxd(b_dq, a_dq), maxd(b_dk, a_dk),
           maxd(b_dv, a_dv));
    // 恢复最终输出为 CLI 选中的路径（上面 A/B 最后一次跑的是 wgmma）。
    run_main_wg(wgmma != 0);
#if defined(FA_WGMMA) && defined(FA_TMA)
    // ---- O37 A/B（D=128）：主 kernel Q/dO cp.async vs 4D-TMA（WGMMA 路径，其余逐字相同）。----
    auto run_main_qd = [&](bool tma) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (tma) {
        if (use_regdq)
          launch_bwd_main_qdtma<128, 64, 32, true>(mg, qmap_main, dmap_main, d_q8, d_qs, d_k8,
                                                   d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                                                   d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                   Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main_qdtma<128, 64, 32, false>(mg, qmap_main, dmap_main, d_q8, d_qs, d_k8,
                                                    d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                                                    d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                    Hkv, scale, (int)causal, ksplit);
      } else {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                   d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                   (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                    d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                    d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                    (int)causal, ksplit);
      }
    };
    auto bench_main_qd = [&](bool tma, float* out) {
      run_main_qd(tma);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_main_qd(tma);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float mcp = 0.f, mtma = 0.f;
    bench_main_qd(false, &mcp);
    bench_main_qd(true, &mtma);
    run_main_qd(false);
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_main_qd(true);
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[O37 A/B] main Q/dO cp.async %.4f ms | tma %.4f ms (%.3fx) | "
           "max_abs(tma-vs-cp) dq/dk/dv=%.3e/%.3e/%.3e\n",
           mcp, mtma, mcp / mtma, maxd(b_dq, a_dq), maxd(b_dk, a_dk), maxd(b_dv, a_dv));
    run_main_wg(wgmma != 0);
    // ---- O41 A/B（D=128）：主 kernel Q/dO-TMA vs Q/dO/K/V 全 TMA（WGMMA 路径，其余逐字相同）。----
    auto run_main_kv = [&](bool kvt) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (kvt) {
        if (use_regdq)
          launch_bwd_main_kvtma<128, 64, 32, true>(mg, qmap_main, dmap_main, kmap_main,
                                                   vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                   d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                   (int)causal, ksplit);
        else
          launch_bwd_main_kvtma<128, 64, 32, false>(mg, qmap_main, dmap_main, kmap_main,
                                                    vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                    d_vs, d_do8, d_dos, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                    scale, (int)causal, ksplit);
      } else {
        run_main_qd(true);
      }
    };
    auto bench_main_kv = [&](bool kvt, float* out) {
      run_main_kv(kvt);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_main_kv(kvt);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float mqd = 0.f, mkv = 0.f;
    bench_main_kv(false, &mqd);
    bench_main_kv(true, &mkv);
    run_main_kv(false);
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_main_kv(true);
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[O41 A/B] main Q/dO-TMA %.4f ms | Q/dO/K/V-TMA %.4f ms (%.3fx) | "
           "max_abs(kvtma-vs-qdtma) dq/dk/dv=%.3e/%.3e/%.3e\n",
           mqd, mkv, mqd / mkv, maxd(b_dq, a_dq), maxd(b_dk, a_dk), maxd(b_dv, a_dv));
    run_main_wg(wgmma != 0);
#endif
  }
#endif

  // ---- O11 A/B（仅 causal, D=128）：LSE O1 原版 vs 镜像配对(单缓冲) vs 镜像配对+cp.async ----
  if (causal && D == 128) {
    auto bench_lse = [&](int which, float* out) {
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) {
        if (which == 0)
          launch_lse<128>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, 1);
        else if (which == 1)
          launch_lse_bal<128, 0>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
        else
          launch_lse_bal<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
      }
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float a = 0.f, b = 0.f, c = 0.f;
    bench_lse(0, &a);
    bench_lse(1, &b);
    bench_lse(2, &c);
    printf("[O11 A/B] lse O1 %.4f ms | bal(单缓冲) %.4f ms (%.3fx) | bal+cpasync %.4f ms "
           "(%.3fx)\n",
           a, b, a / b, c, a / c);
#ifdef FA_WGMMA
    // O9c A/B：wgmma（SW128 + m64n64k32）vs O11 mma 版（同 PIPE=1）。同时核对 LSE 数值。
    auto bench_lse_wgm = [&](int which, float* ms_out) {
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) {
        if (which == 3)
          launch_lse_bal_wgmma<128, 0>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
        else
          launch_lse_bal_wgmma<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
      }
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(ms_out, ev0, ev1));
      *ms_out /= iters;
    };
    float w0 = 0.f, w1 = 0.f;
    std::vector<float> lse_wgm((size_t)B * S * H), lse_ref((size_t)B * S * H);
    bench_lse_wgm(3, &w0);
    bench_lse_wgm(4, &w1);
    // 与 O11 mma 版 LSE 对拍（应逐位相同：同数学、同 rowwise scale 顺序）。
    launch_lse_bal<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
    CUDA_CHECK(cudaMemcpy(lse_ref.data(), d_lse, lse_ref.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    launch_lse_bal_wgmma<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
    CUDA_CHECK(cudaMemcpy(lse_wgm.data(), d_lse, lse_wgm.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double le = 0.0;
    for (size_t i = 0; i < lse_ref.size(); ++i)
      le = std::max(le, (double)std::fabs((double)lse_wgm[i] - (double)lse_ref[i]));
    printf("[O9c A/B] lse mma(bal+cpasync) %.4f ms | wgmma(单缓冲) %.4f ms | wgmma(双缓冲) "
           "%.4f ms (%.3fx) | LSE max_abs(vs mma)=%.3e\n",
           c, w0, w1, c / w1, le);
#endif
#if defined(FA_WGMMA) && defined(FA_TMA)
    // ---- O32 A/B：LSE 的 wgmma+cp.async 版 vs 4D-TMA 版（同 session + 数值对拍）----
    {
      float* d_lse2 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lse2, (size_t)B * S * H * sizeof(float)));
      auto launch_w = [&]() {
        launch_lse_bal_wgmma<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse2, S, H, Hkv, scale);
      };
      auto launch_t = [&]() {
        launch_lse_bal_tma<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
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
      printf("[O32 A/B] lse tma %.4f ms | wgmma %.4f ms (wgmma/tma %.3fx) | "
             "max_abs(tma-vs-wgmma)=%.3e\n", ms_t, ms_w, ms_w / ms_t, e);
      cudaFree(d_lse2);
    }
    // ---- O38 A/B：LSE 的 K 维 split + 二次归约（TMA 版，D=128/causal）----
    //   split=1 逐位退回 O32；split=2/4/8 各扫 1/split 的 K tile 切片后 merge。逐元素对拍
    //   以 split=1 为基准（理论等价、只差 fp32 求和次序）。
    {
      std::vector<float> ref_l((size_t)B * S * H, 0.f);
      float base_ms = 0.f, best = 1e9f;
      int bestk = 1;
      for (int sp = 1; sp <= 8; sp *= 2) {
        auto launch_sp = [&]() {
          launch_lse_bal_tma_split<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                           d_lse_part, S, H, Hkv, scale, sp);
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
      printf("[O38] best lse split=%d %.4f ms (%.3fx)\\n", bestk, best, base_ms / best);
    }
#endif
  }

  // ---- O39 A/B（D=512/MLA/causal）：`lse_mma_kernel_bal<512>` 的 K 维 split + 二次归约 ----
  //   split=1 逐位退回 O11 原路径；split>1 各扫 1/split 的 K tile 切片后 merge。逐元素对拍
  //   以 split=1 为基准（理论等价、只差 fp32 求和次序）。仅 causal（非 causal 走 O1 原版）。
  if (causal && D == 512) {
    std::vector<float> ref_l((size_t)B * S * H, 0.f);
    float base_ms = 0.f, best = 1e9f;
    int bestk = 1;
    for (int sp = 1; sp <= 16; sp *= 2) {
      auto launch_sp = [&]() {
        launch_lse_bal<512, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                               nullptr, d_lse_part, sp);
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
      printf("[O39 A/B] lse(D=512) split=%d %.4f ms (%.3fx vs split1) | max_abs vs split1=%.3e\\n",
             sp, t, base_ms / t, e);
      if (t < best) { best = t; bestk = sp; }
    }
    printf("[O39] best lse split=%d %.4f ms (%.3fx)\\n", bestk, best, base_ms / best);
  }

  // ---- O12 A/B（D=128/512）：主 kernel 的 LSE/D 预装寄存器 ON vs OFF（同 session 计时）。----
  //  OFF 版就是旧的「GEMM1/2 epilogue 里逐元素 global 读 lse/delta」。数值应逐位相同。
  if (D == 128 || D == 512) {
#ifdef FA_WGMMA
    const bool wg_ab = (wgmma != 0);
#else
    const bool wg_ab = false;
#endif
    auto launch_sel = [&](bool prel) {
      if (D == 128) {
        launch128(use_regdq, wg_ab, prel, true);
      } else {
        if (prel)
          launch_bwd_main<512, 64, 32, false, false, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<512, 64, 32, false, false, false>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      }
    };
    auto bench_prel = [&](bool prel, float* out) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_sel(prel);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_sel(prel);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float m_off = 0.f, m_on = 0.f;
    bench_prel(false, &m_off);
    bench_prel(true, &m_on);
    // 逐元素对拍（应逐位相同）
    std::vector<float> p_off_dq(nq), p_on_dq(nq);
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    launch_sel(false);
    CUDA_CHECK(cudaMemcpy(p_off_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    launch_sel(true);
    CUDA_CHECK(cudaMemcpy(p_on_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    double pd = 0.0;
    for (size_t i = 0; i < p_on_dq.size(); ++i)
      pd = std::max(pd, (double)std::fabs((double)p_off_dq[i] - (double)p_on_dq[i]));
    printf("[O12 A/B] main LSE/D preload off %.4f ms | on %.4f ms (%.3fx) | "
           "max_abs(on-vs-off) dq=%.3e\n",
           m_off, m_on, m_off / m_on, pd);
    run_main();  // 恢复 CLI 选中路径（写回 d_dq_acc，不影响 d_dq）
  }

  // ---- O7e-2 A/B（D=128）：fold 的 Ap/dS3/dS2 用 16B 向量化写 vs O7e 的 4B 写 ----
  //   同 session 计时 + 逐元素对拍（同一份 fp8 操作数 ⇒ 应逐位相同）。
  if (D == 128) {
#ifdef FA_WGMMA
    const bool wg_ab = (wgmma != 0);
#else
    const bool wg_ab = false;
#endif
    auto launch_f16b = [&](bool f16) { launch128(use_regdq, wg_ab, prel_sel, f16); };
    auto bench_f16b = [&](bool f16, float* out) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_f16b(f16);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_f16b(f16);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float b4 = 0.f, b16 = 0.f;
    bench_f16b(false, &b4);
    bench_f16b(true, &b16);
    // 逐元素对拍（4B vs 16B，应逐位相同）。
    std::vector<float> x_dq(nq), x_dk(nkv), x_dv(nkv), y_dq(nq), y_dk(nkv), y_dv(nkv);
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_f16b(false);
    CUDA_CHECK(cudaMemcpy(x_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_f16b(true);
    CUDA_CHECK(cudaMemcpy(y_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd2 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O7e-2 A/B] main fold 4B %.4f ms | 16B %.4f ms (%.3fx) | "
           "max_abs(16B-vs-4B) dq/dk/dv=%.3e/%.3e/%.3e\n",
           b4, b16, b4 / b16, maxd2(y_dq, x_dq), maxd2(y_dk, x_dk), maxd2(y_dv, x_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O27 A/B（D=128）：fold 量化 逐元素精确除法 vs 每行 rcp+乘法 ----
  //   `scA/sc3/sc2` 是每输出行一个的常量，原实现让每个元素都发一条精确 fp32 除法
  //   （prec-div，~10+ 指令）；改成每行一次 `__frcp_rn` + 乘法。数学等价，fp8 只有 3 位
  //   尾数 ⇒ cvt 结果几乎不变。同 session 计时 + 逐元素对拍。
  if (D == 128) {
#ifdef FA_WGMMA
    const bool wg_ab = (wgmma != 0);
#else
    const bool wg_ab = false;
#endif
    auto launch_rcp = [&](bool rcp) { launch128(use_regdq, wg_ab, prel_sel, f16b_sel, rcp); };
    auto bench_rcp = [&](bool rcp, float* out) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_rcp(rcp);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_rcp(rcp);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float bdiv = 0.f, brcp = 0.f;
    bench_rcp(false, &bdiv);
    bench_rcp(true, &brcp);
    std::vector<float> x_dq(nq), x_dk(nkv), x_dv(nkv), y_dq(nq), y_dk(nkv), y_dv(nkv);
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_rcp(false);
    CUDA_CHECK(cudaMemcpy(x_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_rcp(true);
    CUDA_CHECK(cudaMemcpy(y_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd2 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O27 A/B] main fold div %.4f ms | rcp-mul %.4f ms (%.3fx) | "
           "max_abs(rcp-vs-div) dq/dk/dv=%.3e/%.3e/%.3e\n",
           bdiv, brcp, bdiv / brcp, maxd2(y_dq, x_dq), maxd2(y_dk, x_dk), maxd2(y_dv, x_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O22 A/B（D=128）：主 kernel 的寄存器 dQ 累加（REGDQ）ON vs OFF（mma 路径）----
  //   O21 的 ncu 显示：REGDQ 的 `dqacc[2][8][4]`（64 个 fp32）把 168-reg 预算挤爆，PTXAS 报
  //   68B spill stores / 380B spill loads，local 占 L1TEX sector 9.57%（Est. 28–50%）。关掉
  //   REGDQ 则 0 spill、且恢复 O3 寄存器预取，代价是 dQ 每个 nt tile 发一次跨 CTA atomicAdd
  //   （归约指令 O(1)→O(ntiles)）。这里同 session 计时 + 逐元素对拍（数学口径相同，仅 dQ 归约
  //   指令数/次序不同；dk/dv 不受影响）。
  if (D == 128) {
    auto launch_reg = [&](bool reg) { launch128(reg, false, prel_sel, f16b_sel); };
    auto bench_reg = [&](bool reg, float* out) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_reg(reg);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_reg(reg);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float ron = 0.f, roff = 0.f;
    bench_reg(true, &ron);
    bench_reg(false, &roff);
    std::vector<float> x_dq(nq), y_dq(nq);
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_reg(true);
    CUDA_CHECK(cudaMemcpy(x_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_reg(false);
    CUDA_CHECK(cudaMemcpy(y_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    double dqdiff = 0.0;
    for (size_t i = 0; i < nq; ++i)
      dqdiff = std::max(dqdiff, std::fabs((double)x_dq[i] - (double)y_dq[i]));
    printf("[O22 A/B] main REGDQ on %.4f ms | off %.4f ms (%.3fx) | max_abs(on-vs-off) dq=%.3e\n",
           ron, roff, ron / roff, dqdiff);
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O14 A/B：输入量化 旧 per-row 标量 vs 新 warp-per-row 向量化（同 session 计时 + 逐位对拍）----
  {
    auto bench_q = [&](bool fast, float* out) {
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) { if (fast) quant_new(); else quant_old(); }
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float qo = 0.f, qn = 0.f;
    bench_q(false, &qo);
    bench_q(true, &qn);
    // 逐字节对拍（q8/qs/k8/ks/v8/vs/do8/dos 应完全相同）。
    std::vector<unsigned char> o_q8(nq), o_k8(nkv), o_v8(nkv), o_do8(nq);
    std::vector<float> o_qs(rows_q), o_ks(rows_kv), o_vs(rows_kv), o_dos(rows_q);
    quant_old();
    CUDA_CHECK(cudaMemcpy(o_q8.data(), d_q8, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_k8.data(), d_k8, nkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_v8.data(), d_v8, nkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_do8.data(), d_do8, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_qs.data(), d_qs, rows_q * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_ks.data(), d_ks, rows_kv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_vs.data(), d_vs, rows_kv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_dos.data(), d_dos, rows_q * 4, cudaMemcpyDeviceToHost));
    quant_new();
    std::vector<unsigned char> n_q8(nq), n_k8(nkv), n_v8(nkv), n_do8(nq);
    std::vector<float> n_qs(rows_q), n_ks(rows_kv), n_vs(rows_kv), n_dos(rows_q);
    CUDA_CHECK(cudaMemcpy(n_q8.data(), d_q8, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_k8.data(), d_k8, nkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_v8.data(), d_v8, nkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_do8.data(), d_do8, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_qs.data(), d_qs, rows_q * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_ks.data(), d_ks, rows_kv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_vs.data(), d_vs, rows_kv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_dos.data(), d_dos, rows_q * 4, cudaMemcpyDeviceToHost));
    long long mismatch = 0;
    auto cmp_u8 = [&](const std::vector<unsigned char>& a, const std::vector<unsigned char>& c) {
      for (size_t i = 0; i < a.size(); ++i) if (a[i] != c[i]) ++mismatch;
    };
    auto cmp_f = [&](const std::vector<float>& a, const std::vector<float>& c) {
      for (size_t i = 0; i < a.size(); ++i) if (a[i] != c[i]) ++mismatch;
    };
    cmp_u8(o_q8, n_q8); cmp_u8(o_k8, n_k8); cmp_u8(o_v8, n_v8); cmp_u8(o_do8, n_do8);
    cmp_f(o_qs, n_qs); cmp_f(o_ks, n_ks); cmp_f(o_vs, n_vs); cmp_f(o_dos, n_dos);
    printf("[O14 A/B] quant old(per-row) %.4f ms | new(warp-per-row) %.4f ms (%.3fx) | "
           "bitwise mismatch=%lld\n", qo, qn, qo / qn, mismatch);
    run_all();  // 恢复 CLI 选中路径的完整输出
  }

  // ---- O26 A/B：delta 旧 per-row(smem 归约) vs 新 warp-per-row 向量化（同 session 计时 + 对拍）----
  {
    auto bench_delta = [&](bool warp, float* out) {
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) {
        if (warp) {
          if (D == 128)
            delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
          else
            delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        } else {
          if (D == 128) delta_kernel<128><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
          else delta_kernel<512><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
        }
      }
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float do_ms = 0.f, dw_ms = 0.f;
    bench_delta(false, &do_ms);
    bench_delta(true, &dw_ms);
    if (D == 128) delta_kernel<128><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    else delta_kernel<512><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    std::vector<float> d_old(rows_q);
    CUDA_CHECK(cudaMemcpy(d_old.data(), d_delta, rows_q * 4, cudaMemcpyDeviceToHost));
    if (D == 128)
      delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    else
      delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    std::vector<float> d_new(rows_q);
    CUDA_CHECK(cudaMemcpy(d_new.data(), d_delta, rows_q * 4, cudaMemcpyDeviceToHost));
    float md = 0.f, mr = 0.f;
    for (size_t i = 0; i < d_old.size(); ++i) {
      float ad = fabsf(d_old[i] - d_new[i]);
      if (ad > md) md = ad;
      float den = fmaxf(fabsf(d_old[i]), 1e-6f);
      mr = fmaxf(mr, ad / den);
    }
    printf("[O26 A/B] delta old(per-row) %.4f ms | new(warp-per-row) %.4f ms (%.3fx) | "
           "max_abs=%.3e max_rel=%.3e\n", do_ms, dw_ms, do_ms / dw_ms, md, mr);
    run_all();  // 恢复 CLI 选中路径的完整输出
  }

  // ---- O19 A/B（D=128）：主 kernel mma(BM=64,3 CTA/SM) vs wg2(BM=128,2 wg,256 线程,1 CTA/SM)
  //      同一 session 计时 + 逐元素对拍（只改归约结构/几何，数学口径不变）。----
  if (D == 128) {
    auto run_mma_sel = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (use_regdq)
        launch_bwd_main<128, 64, 32, true, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      else
        launch_bwd_main<128, 64, 32, false, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
    };
    auto run_wg2_sel = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_wg2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                          d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                          ksplit2);
    };
    auto bench_sel = [&](auto&& fn, float* out) {
      fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float t_mma = 0.f, t_wg2 = 0.f;
    bench_sel(run_mma_sel, &t_mma);
    bench_sel(run_wg2_sel, &t_wg2);
    std::vector<float> a_dq(nq), a_dk(nkv), a_dv(nkv), b_dq(nq), b_dk(nkv), b_dv(nkv);
    run_mma_sel();
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_wg2_sel();
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O19 A/B] main mma(BM64,grid=%d) %.4f ms | wg2(BM128,grid=%d) %.4f ms (%.3fx) | "
           "max_abs(wg2-vs-mma) dq/dk/dv=%.3e/%.3e/%.3e\n",
           mg.x, t_mma, mg2.x, t_wg2, t_mma / t_wg2, maxd(b_dq, a_dq), maxd(b_dk, a_dk),
           maxd(b_dv, a_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O21 A/B（D=128）：主 kernel KV tile BN=32（3 CTA/SM）vs BN=64（2 CTA/SM，tile 数减半）。
  //      同一 session 计时 + 逐元素对拍（只改 KV 分块宽度，数学口径不变）。----
  if (D == 128) {
    auto run_bn = [&](int bn) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (bn == 64) {
        if (use_regdq)
          launch_bwd_main<128, 64, 64, true, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 64, false, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      } else {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      }
    };
    auto bench_bn = [&](int bn, float* out) {
      run_bn(bn);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_bn(bn);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float t32 = 0.f, t64 = 0.f;
    bench_bn(32, &t32);
    bench_bn(64, &t64);
    std::vector<float> a_dq(nq), a_dk(nkv), a_dv(nkv), b_dq(nq), b_dk(nkv), b_dv(nkv);
    run_bn(32);
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_bn(64);
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O21 A/B] main BN=32 %.4f ms | BN=64 %.4f ms (%.3fx) | "
           "max_abs(BN64-vs-BN32) dq/dk/dv=%.3e/%.3e/%.3e\n",
           t32, t64, t32 / t64, maxd(b_dq, a_dq), maxd(b_dk, a_dk), maxd(b_dv, a_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O21b A/B：冗余 fp32→fp32 convert 的代价（默认 cvt_on=0 已消掉）。acc 已别名到输出，
  //      `do_cvt=true` 即对同一缓冲做一次自拷贝 ⇒ 时间差就是 convert 的真实成本。----
  {
    auto run_pipe = [&](bool do_cvt) {
      quant();
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      run_preprocess();
      run_main();
      if (do_cvt)
        convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                    d_dv, nq, nkv);
    };
    auto bench_pipe = [&](bool do_cvt, float* out) {
      run_pipe(do_cvt);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_pipe(do_cvt);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float tt_on = 0.f, tt_off = 0.f;
    bench_pipe(true, &tt_on);
    bench_pipe(false, &tt_off);
    printf("[O21b A/B] end2end 保留convert %.4f ms | 直写输出(消convert) %.4f ms (%.4fx)\n",
           tt_on, tt_off, tt_on / tt_off);
    run_all();  // 恢复 CLI 选中路径
  }

  std::vector<float> mdq(nq), mdk(nkv), mdv(nkv);
  CUDA_CHECK(cudaMemcpy(mdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));

  auto print_cmp = [&](const char* name, const std::vector<float>& mine,
                       const std::vector<float>& ref) {
    DiffStat st = diff_stat(mine, ref);
    printf("  %-4s vs ref: max_abs=%.3e  max_rel=%.3e\n", name, st.max_abs, st.max_rel);
  };
  printf("[compare] ours vs fp32 ref (O from %s.npy)\n", o_name.c_str());
  print_cmp("dq", mdq, rdq.data);
  print_cmp("dk", mdk, rdk.data);
  print_cmp("dv", mdv, rdv.data);
  // P5-3 调试：head_dim>128 时按每 128 维一段给 max_abs，验证 GEMM3/4/5 的 N-tile
  // 循环每一段都正确（若某段漏算/错位，该段误差会 O(1) 明显大于其它段）。
  if (D > 128) {
    auto print_band = [&](const char* name, const std::vector<float>& mine,
                          const std::vector<float>& ref) {
      for (int bnd = 0; bnd < D / 128; ++bnd) {
        double mx = 0.0;
        for (size_t i = 0; i < mine.size(); ++i)
          if ((int)(i % (size_t)D) / 128 == bnd)
            mx = std::max(mx, std::fabs((double)mine[i] - (double)ref[i]));
        printf("    %s[d %3d..%3d] max_abs=%.3e\n", name, bnd * 128, bnd * 128 + 127, mx);
      }
    };
    print_band("dq", mdq, rdq.data);
    print_band("dk", mdk, rdk.data);
    print_band("dv", mdv, rdv.data);
  }

  auto cmp_to = [&](const char* name, const std::vector<float>& mine, const char* fname) {
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
  // O21b：acc 指针已别名到 dq/dk/dv，不能重复 free。
  cudaFree(d_dq); cudaFree(d_dk); cudaFree(d_dv);
  return 0;
}
