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
#include "../fa_bwd_dump.h"   // P3-3：ours 输出落 npy，供 harness/fa_bwd_compare.py

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

// 第 162 轮 O68：非 causal（full）D=128 的 LSE 用 O54 的均衡版
//   `lse_mma_kernel_bal<HD,PIPE,FULL=true>`（一个 CTA 一个 m 块 + cp.async 双缓冲）
//   替代 O1 的 `lse_mma_kernel`（无流水，逐标量 global→smem）。默认 1；`--lsefull=0`
//   退回 O1 做同 binary A/B。定长与 varlen 两条 full 路径都读这个开关（故用文件作用域）。
static int g_lse_full_opt = 1;
// 第 166 轮 O72：varlen full D=128 的 LSE 是否走 4D-TMA（对齐定长 O70）。默认 1；`--lsetmavarlen=0`
//   退回 O68 的 cp.async 均衡版做同 binary A/B。需要 `-DFA_WGMMA -DFA_TMA` 构建（否则恒 0）。
static int g_lse_tma_varlen = 1;
// O101（第 195 轮）：**定长 full 的 LSE 补齐均衡/流水/split**——此前 D=512（MLA）定长 full 的
//   LSE 一直是 O1 的 `lse_mma_kernel`（逐标量 global→smem、无 cp.async 流水、无 K 维 split），
//   与 O54 只修了 varlen full 形成不对称；实测该 LSE 占 full MLA 端到端 ~65%（S1024H2：
//   preprocess 0.376ms > main 0.178ms）。默认 1 = 走 O54 的 `lse_mma_kernel_bal<512,1,true>`
//   （一个 CTA 一个 m 块 + cp.async 双缓冲 + K 维 split）；`--lse512old=1` 退回 O1 做同 binary A/B。
static int g_lse_full512_opt = 1;
// F6-③（O77）：4D-TMA 描述符的 L2 promotion 档位（0=NONE/1=L2_128B/2=L2_256B）。
// 定义在 `FA_TMA` 守卫外，好让非 TMA（sm_90/mma）构建也能编译（值不被使用）。
static int g_l2promo = 0;

#if defined(FA_WGMMA) && defined(FA_TMA)
// O32：为 LSE 的 Q/K 建 4D TMA 描述符（dims={D,S,H,B}，SW128，box={128,64,1,1}）。
// fp8 一行 = 128 字节 = SW128 atom 整行 ⇒ 一个 box 覆盖整个 head_dim（不像 fp16 需 2 chunk）。
// dtype 用 UINT8（CUDA 13 驱动枚举无 FLOAT8_E4M3）。globalStride（字节）：dim1(S) 行距 = H*D，
// dim2(H) 头距 = D，dim3(B) 批距 = S*H*D（元素即字节）。要求 16B 对齐（D=128 恒成立）。
// F6-③（O77）：Q/dO/K/V 的 4D-TMA 描述符的 L2 promotion 档位（见上，定义在守卫外）。
static CUtensorMap make_lse_map_fp8(const void* ptr, long long H, long long S, long long D,
                                    long long B, uint32_t boxR = 64) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D), (uint64_t)D, (uint64_t)(S * H * D)};
  uint32_t box[4] = {128, boxR, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  const CUtensorMapL2promotion promo =
      (g_l2promo == 2) ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B
                       : (g_l2promo == 1) ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B
                                          : CU_TENSOR_MAP_L2_PROMOTION_NONE;
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", s);
    std::exit(1);
  }
  return map;
}

// O78：为「按 head 分块」的 overlap 建**子范围**描述符——head 计数 `hc`（dims[2]）与
//   全张量 head 数 `Hfull`（stride 用）解耦，基址前移 `h0*D`。这样 kernel 里 head 坐标
//   仍取 `blockIdx.y∈[0,hc)`，而物理行距仍是全 head 数 ⇒ 与全量路径逐元素同址。
static CUtensorMap make_map_fp8_chunk(const void* ptr, long long Hfull, long long h0, long long hc,
                                      long long S, long long D, long long B, uint32_t boxR) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)hc, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(Hfull * D), (uint64_t)D, (uint64_t)(S * Hfull * D)};
  uint32_t box[4] = {128, boxR, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  const char* base = reinterpret_cast<const char*>(ptr) + (size_t)h0 * D;
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, (void*)base, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled(chunk) failed: %s\n", s);
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
// O47：`NTH`/`NWAR` 透传给主 kernel（默认 128/2 与历史逐字等价；MLA 用 256/4）。
// O104（第 198 轮）：`HSWAP`（默认 false）同 `fa_bwd_fp8_mma_kernel`——把 O93 的跨 head 全局 LPT
//   调度推广到通用 mma/wgmma 主 kernel（D=256 默认走它）。`HSWAP=false` 与历史逐位相同。
template <int HD, int BM, int BN, bool REGDQ, bool WGMMA = false, bool PREL = true, bool F16B = true,
          bool RCP = true, int NTH = THREADS, int NWAR = WN, bool HSWAP = false>
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
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, NTH, NWAR, false, false,
                            false, false, HSWAP>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, NTH, NWAR, false, false,
                        false, false, HSWAP><<<mg, NTH, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit, cu_seqlens, mt_b, mt_m);
}

// P3-4e/P3-4f：确定性 dK/dV/dQ 版主 kernel（`DET=true`）。把 dK/dV 的跨 CTA `atomicAdd` 换成
//   「按 (Q 头, Q 块) 分片的 partial 覆盖写 + `dkv_reduce_kernel` 固定次序求和」；`ksplit>1` 时
//   dQ 也走 partial（`dq_part`，按 part 分片）+ `dq_reduce_kernel`。仅用于定长（非 varlen）、
//   默认 mma 路径（A/B 实验，见 docs/03 §45/§57）。
//   注：`ksplit>1` 时 kRegDq 路径（HD=128）先在寄存器累加再**覆盖写** partial；非 kRegDq
//   路径（MLA/HD=512，P3-4k）逐 tile **累加进**本 CTA 独占的 partial 区（同一 (row,c) 由同一
//   线程按 nt 程序序写 ⇒ 确定），两路都无需再要求 `REGDQ=true`。`ksplit==1` 时 dQ 仍走无竞争
//   `red_add2`（与 P3-4e 逐位不变）。
// P3-4h：`NTH`/`NWAR` 参数化（默认 THREADS/WN ⇒ D=128 路径逐字不变），好让 MLA（HD=512）
//   复用同一 DET 路径、与默认 8-warp/256 几何（O47/O51）同几何做 A/B。
// P3-4i：`WGMMA`（默认 false，定长路径逐字不变）让 varlen D=128（varlen 构建恒为
//   `-DFA_WGMMA`，默认主 kernel 走 WGMMA）能用同一 DET 路径；`cu_seqlens/mt_b/mt_m` 透传
//   varlen 的 packed 索引（定长不传 ⇒ 逐字不变）。
// P3-4m：`KVPIPE` 让 DET 的 MLA 主 kernel 也能用 O51 的 K/V `cp.async` 回填流水（device 的
//   `fp8_mma_body` 早已同时支持 `KVPIPE && DET`，此前 host 未接线 ⇒ MLA DET 一直走非 kvpipe
//   主 kernel，白扔 O51 的 1.78–1.86×）。数值与旧非 kvpipe DET 逐位相同（只改搬运）。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          int NTH = THREADS, int NWAR = WN, bool WGMMA = false, bool KVPIPE = false,
          bool DET_HALF = false>
static void launch_bwd_main_det(dim3 mg, const unsigned char* q8, const float* qs,
                                const unsigned char* k8, const float* ks,
                                const unsigned char* v8, const float* vs,
                                const unsigned char* do8, const float* dos,
                                const float* delta, const float* lse, float* dq_acc,
                                float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                float scale, int causal, int ksplit, float* dk_part,
                                float* dv_part, int nblk, float* dq_part = nullptr,
                                const int* cu_seqlens = nullptr, const int* mt_b = nullptr,
                                const int* mt_m = nullptr, const int* part_base = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = WGMMA ? Cfg::smem_bytes_wgmma
                              : (KVPIPE ? Cfg::smem_bytes_kvpipe : Cfg::smem_bytes);
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, NTH, NWAR, KVPIPE, true,
                            DET_HALF>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, NTH, NWAR, KVPIPE, true,
                        DET_HALF>
      <<<mg, NTH, kSmem>>>(q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
                           dv_acc, S, H, Hkv, scale, causal, ksplit, cu_seqlens, mt_b,
                           mt_m, dk_part, dv_part, nblk, dq_part, part_base);
}

// O51：K/V `cp.async` 回填流水版主 kernel（mma 后端，MLA/HD=512）。smem 比 `launch_bwd_main`
//   多 `2*BN*QTS`（Ap/dS3 独立缓冲），K/V 在 GEMM1/2 后异步回填下一 tile。数值与旧路径逐位相同。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          int NTH = THREADS, int NWAR = WN>
static void launch_bwd_main_kvpipe(dim3 mg, const unsigned char* q8, const float* qs,
                                   const unsigned char* k8, const float* ks,
                                   const unsigned char* v8, const float* vs,
                                   const unsigned char* do8, const float* dos,
                                   const float* delta, const float* lse, float* dq_acc,
                                   float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                   float scale, int causal, int ksplit,
                                   const int* cu_seqlens = nullptr,
                                   const int* mt_b = nullptr, const int* mt_m = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_kvpipe;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, false, PREL, F16B, RCP, NTH, NWAR, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, false, PREL, F16B, RCP, NTH, NWAR, true>
      <<<mg, NTH, kSmem>>>(q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
                           dv_acc, S, H, Hkv, scale, causal, ksplit, cu_seqlens, mt_b, mt_m);
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
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          bool HSWAP = false>
static void launch_bwd_main_kvtma(dim3 mg, const CUtensorMap& qmap, const CUtensorMap& dmap,
                                  const CUtensorMap& kmap, const CUtensorMap& vmap,
                                  const unsigned char* q8, const float* qs,
                                  const unsigned char* k8, const float* ks,
                                  const unsigned char* v8, const float* vs,
                                  const unsigned char* do8, const float* dos,
                                  const float* delta, const float* lse, float* dq_acc,
                                  float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                  float scale, int causal, int ksplit,
                                  cudaStream_t st = nullptr, const int* mt_m = nullptr,
                                  const int* slot_tab = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_wgmma_kvtma;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, false, false, false, HSWAP>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, false, false, false, HSWAP>
      <<<mg, THREADS, kSmem, st>>>(qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos,
                               delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal,
                               ksplit, nullptr, nullptr, nullptr, 0, nullptr, nullptr, mt_m,
                               slot_tab);
}

// P3-4g：把 `--det` 从默认 mma 路径扩到 Hopper TMA 快路（`launch_bwd_main_kvtma` 的
//   `DET=true` 版）。dK/dV 走 partial + `dkv_reduce_kernel` 固定次序归约（可复现）。
//   P3-4g 原版锁 `ksplit=1`：dQ 由唯一 CTA 的非原子 `red_add2` 写（无竞争）⇒ 确定；
//   **O102（第 196 轮）**放开 `ksplit>1`：`DET && ksplit>1` 时 dQ 写 `dq_part[((row*H+h)*ksplit
//   + part)*HD+c]` 分片 partial（同 mma 路径 P3-4f），host 侧 `dq_reduce_kernel` 按 part 固定
//   次序求和 ⇒ dQ/dK/dV 仍逐位可复现，同时恢复 split-K 并行度。`dq_part` 默认 nullptr（ksplit=1
//   ⇒ 逐位退化为 P3-4g 原路径）。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          bool DET_HALF = false>
static void launch_bwd_main_kvtma_det(dim3 mg, const CUtensorMap& qmap, const CUtensorMap& dmap,
                                      const CUtensorMap& kmap, const CUtensorMap& vmap,
                                      const unsigned char* q8, const float* qs,
                                      const unsigned char* k8, const float* ks,
                                      const unsigned char* v8, const float* vs,
                                      const unsigned char* do8, const float* dos,
                                      const float* delta, const float* lse, float* dq_acc,
                                      float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                      float scale, int causal, float* dk_part, float* dv_part,
                                      int nblk, int ksplit = 1, float* dq_part = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_wgmma_kvtma;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, true, DET_HALF>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, true, DET_HALF>
      <<<mg, THREADS, kSmem>>>(qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos,
                               delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal,
                               ksplit, nullptr, dk_part, dv_part, nblk, dq_part);
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

// F6 第二步：双 warpgroup + wgmma（GEMM1/2）的 BM=128 主 kernel。仅 `-DFA_WGMMA` 构建可用。
#ifdef FA_WGMMA
template <int HD>
static constexpr int wgmma2_smem_bytes() {
  constexpr int BM = 128, BN = 32;
  constexpr int PSLD = HD + 8, QTS = BM + 16, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024, KS_SZ = (BN / 8) * (HD / 128) * 1024;
  constexpr int qp_bytes = (BM / 2) * PSLD * 2, kp_bytes = (BN / 2) * PSLD * 2;
  // 前 4 块 SW128（Q/dO/K/V）+ Qp/dOp/Kp + dS2 + scales/Ps/Ss + 独立 Ap/dS3 + 1024B 对齐 slack。
  return 2 * QS_SZ + 2 * KS_SZ + 2 * qp_bytes + kp_bytes + BM * DSS2 +
         (kNScale + 2 * BM * PSS) * (int)sizeof(float) + 2 * BN * QTS + 1024;
}

template <int HD>
static void launch_bwd_wgmma2(dim3 mg, const unsigned char* q8, const float* qs,
                              const unsigned char* k8, const float* ks,
                              const unsigned char* v8, const float* vs,
                              const unsigned char* do8, const float* dos,
                              const float* delta, const float* lse, float* dq_acc,
                              float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                              float scale, int causal, int ksplit) {
  constexpr int kSmem = wgmma2_smem_bytes<HD>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wgmma2_kernel<HD, 128, 32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wgmma2_kernel<HD, 128, 32><<<mg, 256, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}

// O91（F6-step4，multi-warpgroup）：把 `fa_bwd_fp8_wgmma2_kernel` 泛化到 `NWG` 个 warpgroup、
//   `BM = NWG*64`（默认 NWG=2 即原 wgmma2）。`--wg3` 走 `NWG=3`（BM=192、384 线程、1 CTA/SM）：
//   每 KV 元素被贡献的 CTA 数按 m 块数 S/BM 再降 1.5×（vs BM=128），且 12 warp/SM（vs 8）摊延迟。
template <int HD, int BM, int BN, int NWG>
static constexpr int wgmma_nw_smem_bytes() {
  constexpr int PSLD = HD + 8, QTS = BM + 16, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024, KS_SZ = (BN / 8) * (HD / 128) * 1024;
  constexpr int qp_bytes = (BM / 2) * PSLD * 2, kp_bytes = (BN / 2) * PSLD * 2;
  return 2 * QS_SZ + 2 * KS_SZ + 2 * qp_bytes + kp_bytes + BM * DSS2 +
         (kNScale + 2 * BM * PSS) * (int)sizeof(float) + 2 * BN * QTS + 1024;
}

template <int HD, int BM, int BN, int NWG>
static void launch_bwd_wgmma_nw(dim3 mg, const unsigned char* q8, const float* qs,
                                const unsigned char* k8, const float* ks,
                                const unsigned char* v8, const float* vs,
                                const unsigned char* do8, const float* dos,
                                const float* delta, const float* lse, float* dq_acc,
                                float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                float scale, int causal, int ksplit) {
  constexpr int kSmem = wgmma_nw_smem_bytes<HD, BM, BN, NWG>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wgmma2_kernel<HD, BM, BN, NWG>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wgmma2_kernel<HD, BM, BN, NWG><<<mg, NWG * 128, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}

// O90（F6-step3）：wgmma2 + K/V 4D-TMA（K 双缓冲）的 launcher。需 `-DFA_WGMMA -DFA_TMA`。
#if defined(FA_WGMMA) && defined(FA_TMA)
template <int HD>
static constexpr int wgmma2tma_smem_bytes() {
  constexpr int BM = 128, BN = 32;
  constexpr int PSLD = HD + 8, QTS = BM + 16, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024, KS_SZ = (BN / 8) * (HD / 128) * 1024;
  constexpr int qp_bytes = (BM / 2) * PSLD * 2, kp_bytes = (BN / 2) * PSLD * 2;
  // 2 块 Q/dO SW128 + K 双缓冲 + V + Qp/dOp/Kp + dS2 + scales/Ps/Ss + Ap/dS3 + bars(64) + slack。
  return 2 * QS_SZ + 3 * KS_SZ + 2 * qp_bytes + kp_bytes + BM * DSS2 +
         (kNScale + 2 * BM * PSS) * (int)sizeof(float) + 2 * BN * QTS + 64 + 1024;
}

template <int HD>
static void launch_bwd_wgmma2tma(dim3 mg, const CUtensorMap& kmap, const CUtensorMap& vmap,
                                 const unsigned char* q8, const float* qs,
                                 const unsigned char* k8, const float* ks,
                                 const unsigned char* v8, const float* vs,
                                 const unsigned char* do8, const float* dos,
                                 const float* delta, const float* lse, float* dq_acc,
                                 float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                 float scale, int causal, int ksplit) {
  (void)k8; (void)v8;   // K/V 由 4D-TMA 搬入
  constexpr int kSmem = wgmma2tma_smem_bytes<HD>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wgmma2_tma_kernel<HD, 128, 32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wgmma2_tma_kernel<HD, 128, 32><<<mg, 256, kSmem>>>(
      kmap, vmap, q8, qs, ks, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}
#endif  // FA_WGMMA && FA_TMA
#endif  // FA_WGMMA

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
//   O54：`FULL=true` 时也服务非 causal（full）——一个 CTA 一个 m 块（lg.x=nblk），无镜像配对。
//   O58：模板参数化为 `<HD,PIPE,FULL,NTH,LBN_>`（对齐 device 侧；默认档 `NTH=THREADS,LBN_=64`
//   与原版逐位等价）。smem 按 `LBM_=(NTH/32)*16`、`LBN_` 现算。
template <int HD, int PIPE, bool FULL = false, int NTH = THREADS, int LBN_ = 64>
static void launch_lse_bal(dim3 lg, const unsigned char* q8, const float* qs,
                           const unsigned char* k8, const float* ks, float* lse, int S, int H,
                           int Hkv, float scale, const int* cu = nullptr,
                           float* lse_part = nullptr, int ksplit = 1,
                           long long merge_rows = -1) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int ASLD = Cfg::ASLD;
  constexpr int LBM_ = (NTH / 32) * 16;
  constexpr int kSmem = LBM_ * ASLD + (PIPE ? 2 : 1) * LBN_ * ASLD +
                        (LBM_ + (PIPE ? 2 : 1) * LBN_) * (int)sizeof(float);
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<HD, PIPE, FULL, NTH, LBN_>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  // O39：K 维 split —— grid.z 由 B 扩成 B*ksplit，部分结果写 lse_part 后由 merge kernel 汇总。
  const int B = (int)lg.z;
  dim3 g(lg.x, lg.y, (unsigned)((size_t)B * ksplit));
  lse_mma_kernel_bal<HD, PIPE, FULL, NTH, LBN_><<<g, NTH, kSmem>>>(
      q8, qs, k8, ks, lse, S, H, Hkv, scale, cu, lse_part, ksplit);
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
//   O70：模板加 `bool FULL`，`FULL=true` 时 grid.x 应是 `nblk`（一个 m 块一个 CTA，非因果）。
#if defined(FA_WGMMA) && defined(FA_TMA)
//   O72（第 166 轮）：加 `const int* cu`——varlen full D=128 复用它（描述符建在 packed 张量上、
//   行坐标由内核用 `cu_seqlens[b]` 定界）。定长调用不传 ⇒ 逐位不变。
template <int HD, int PIPE, bool FULL = false>
static void launch_lse_bal_tma(dim3 lg, const CUtensorMap& qmap, const CUtensorMap& kmap,
                               const float* qs, const float* ks, float* lse, int S, int H,
                               int Hkv, float scale, const int* cu = nullptr,
                               cudaStream_t st = nullptr) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = Cfg::lse_smem_bytes_tma1;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<HD, PIPE, FULL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  lse_mma_kernel_bal_tma<HD, PIPE, FULL><<<lg, THREADS, kSmem, st>>>(qmap, kmap, qs, ks, lse,
                                                                 nullptr, S, H, Hkv, scale, 1, cu);
}

// O38：LSE 的 K 维 split + 二次归约（仅 D=128/causal/TMA）。`lg.z` 是 batch B；内部把
//   grid.z 扩成 `B*ksplit`，kernel 每个 (pair,ks) 只扫本 m 块的第 ks 个 K tile 切片，把
//   部分 (m,l) 写入 `lse_part`（[B*S*H][ksplit] 个 (m,l)），随后 merge kernel 汇总成 `lse`。
//   `ksplit==1` 时等于原 O32 路径（kernel 内逐位退化，不写 part、不 launch merge）。
//   好处：小 S / 低 H 时镜像配对后的 grid（=`ceil(nblk/2)*H`）常 < 132 SM，split 直接补满
//   并发槽、缩短「单 CTA 顺序扫 nblk+1 个 tile」的临界路径；大 S 已铺满一个波时收益趋零。
//   O103（第 197 轮）：模板加 `bool FULL`——full D=256 也走 TMA+split（此前只有 mma/cp.async
//   版）；FULL=false（causal 镜像配对）时与历史逐字相同。
template <int HD, int PIPE, bool FULL = false>
static void launch_lse_bal_tma_split(dim3 lg, const CUtensorMap& qmap, const CUtensorMap& kmap,
                                     const float* qs, const float* ks, float* lse,
                                     float* lse_part, int S, int H, int Hkv, float scale,
                                     int ksplit, cudaStream_t st = nullptr) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = Cfg::lse_smem_bytes_tma1;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<HD, PIPE, FULL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const int B = (int)lg.z;
  dim3 g(lg.x, lg.y, (unsigned)(B * ksplit));
  lse_mma_kernel_bal_tma<HD, PIPE, FULL><<<g, THREADS, kSmem, st>>>(qmap, kmap, qs, ks, lse, lse_part,
                                                          S, H, Hkv, scale, ksplit);
  if (ksplit > 1) {
    const long long nrows = (long long)B * S * H;
    const int th = 256;
    const long long bl = (nrows + th - 1) / th;
    lse_split_merge_kernel<<<(unsigned)bl, th, 0, st>>>(lse_part, lse, nrows, ksplit);
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
                      bool lse_compact = false, int lse_split = 0, int mla8w = -1,
                      int mla_kvp = -1, int lseocc = 0, int lse8w = 0,
                      const std::string& dump = "", int det_ab = 0, int det_ksplit = 1,
                       int fuse_reduce = 1, int part_compact = 0, int qfuseflag = 1,
                       int dfuseflag = 1, int vksplit = -1, int mrev_flag = 1) {
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
  long auto_k = kp;
  // O98（第 192 轮）：**full（非 causal）变长的 ksplit 重标定**——把 O96/O97 的定长 full
  //   审计推广到 `run_varlen`。原自动档沿用 O29 的 **causal** 标定 target（D=128 用 8192、
  //   D=256/512 用 `maxlen/2`），对 full 过切/欠切；且 `target/base` 含**短序列的早退死
  //   CTA**（`base_grid` 按 `maxlen` 计）⇒ 名义网格被高估。本轮对 dump 的全部 6 个 varlen
  //   full shape 做 k∈[1,16] 全扫（docs/03 §120）：
  //   · **D=128**：O98 取「`max(kp,3)`」下限 3（当时测 b4_t3840 最优 4）；**O100（第 194 轮）
  //     在干净机器上重复实测**（k∈[2,5]×3 次、150 iters）——4 个 shape 的最优**一致为 k=3**
  //     （b4_t4096 1.114 vs k4 1.128、b4_t3840 1.387 vs k4 1.391、b5 2.690、b8 1.129），
  //     O98 的「b4_t3840 k3 反慢 3.9%」证实为当轮噪声/残留进程。故 D=128 full 变长**直接取
  //     k=3**（按 nblk 封顶）——full 无三角偏斜，k 越大只增 Q/dO 重读，实测 3 即最优点。
  //   · **D=512（MLA，1 CTA/SM→132 槽）**：按 O97 的 132 槽波对齐（b1_t512 最优 k=8、
  //     b3_t1792 k=11 恰整数波），实测 +8.6%/+1.8%。
  //   · **D=256**（无 varlen dump，沿用 O97 定长规则：`maxlen≥2048` 给足并发、否则波对齐）。
  //   `--ksplit=K` 显式给出时不覆盖。
  if (!causal) {
    if (D == 128) {
      const long nblk32 = (long)((maxlen + 31) / 32);  // main kernel BN=32
      auto_k = std::min(3L, nblk32);
      if (auto_k < 1) auto_k = 1;
    } else {
      const long SLOTS = (D == 512) ? 132L : 264L;
      const long KMAX = (D == 512) ? 16L : 12L;
      if (D == 256 && maxlen >= 2048) {
        long k = 8192L / base_grid;
        if (k < 1) k = 1;
        if (k > KMAX) k = KMAX;
        auto_k = k;
      } else {
        long best_k = 1, best_waste = -1;
        for (long k = 1; k <= KMAX; ++k) {
          const long g = base_grid * k;
          const long waste = ((g + SLOTS - 1) / SLOTS) * SLOTS - g;
          if (best_waste < 0 || waste < best_waste) { best_waste = waste; best_k = k; }
        }
        if (D == 512 && best_k < 2) best_k = 2;  // 长 K 循环仍偏好 ≥2 份并发（同 O97）
        auto_k = best_k;
      }
    }
  }
  // O99（第 193 轮）：**causal D=256 的 ksplit 重标定**（定长规则同源，用 `maxlen` 当串长）。
  //   与 O98 的 full 分支并列：定长 causal D=256 的 `target=S/2` 是小/中 S 严重欠切（见定长
  //   路径注释 / docs/03 §121），变长沿用同一错；改为 `k = clamp(2*maxlen/base,1,16)` 再按
  //   `nblk(=`maxlen/64`)` 封顶。**当前无 D=256 变长 dump，本分支按定长实测外推、未单独测量**；
  //   只改跨 CTA atomicAdd 次序、数值逐位不变。full 走上面的 O98 分支。
  if (causal && D == 256) {
    const long nblk256 = (long)((maxlen + BM - 1) / BM);
    long k = (2L * maxlen) / base_grid;
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    if (k > nblk256) k = nblk256;
    auto_k = k;
  }
  // O97：`--ksplit=K` 也可用于 varlen（同 binary A/B；K>=1 直接覆盖自动档）。默认 -1 自动。
  const int ksplit = (vksplit >= 1) ? vksplit : (int)auto_k;
  const bool use_regdq = (D == 128) && ((long)(maxlen / 32) / (causal ? 2 : 1) / ksplit >= 4);

  // O106（第 200 轮）：把 O89/O105 的**per-head LPT m 块反转**（`mblk = nblk-1-mt`）推广到
  //   **变长（varlen）causal 主 kernel**。同一序列内 K 循环长 ∝ `ceil(len_b/BM)`，causal 下
  //   m 越大越贵；默认非紧凑网格直接把 `mt = blockIdx.x/ksplit` 当 mblk，是 LPT 的反面（便宜块
  //   先派发、贵块压尾波）。LSE 早已按降序工作量排序（镜像对表），主 kernel 一直没接。只改
  //   「哪个 CTA 算哪个 m 块」，dK/dV 仍是可交换的跨 CTA `atomicAdd` ⇒ 数值仅在 fp8 噪声内。
  //   仅 causal、D==128/512、`nblk_max>=16` 生效（非紧凑网格；`--compact` 走 mt 表本身，不叠加）；
  //   `--mrev=0` 回退历史锯齿序。
  int* d_mrev_v = nullptr;
  const int nblk_mv = (maxlen + BM - 1) / BM;
  //   形状门控（关键，同 O104 hswap256）：只在**名义网格被短序列 padding**（`total_mt < nblk_max*B`，
  //   即存在早退死 CTA、序列长度不齐）时为正——4 个 D=128 shape 实测 b4_t3840 +1.6% / b8_t2904
  //   +1.5% / b5_t3968 +0.6%（均不齐、maxlen=2048），而**等长 b4_t4096（无 padding）为 −1.1%**
  //   （纯 LPT 打乱同序列相邻 m 的 K/V 微局部性、收益不抵）；D=512 b3_t1792 +7.1%。故门控在
  //   「有 padding」；等长档逐值不变。
  const bool mrev_v_elig = (mrev_flag != 0) && causal && (D == 128 || D == 512) &&
                           nblk_mv >= 16 && (long)total_mt < (long)nblk_mv * B;
  if (mrev_v_elig) {
    std::vector<int> hrev(nblk_mv);
    for (int i = 0; i < nblk_mv; ++i) hrev[i] = nblk_mv - 1 - i;
    CUDA_CHECK(cudaMalloc(&d_mrev_v, nblk_mv * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mrev_v, hrev.data(), nblk_mv * sizeof(int), cudaMemcpyHostToDevice));
    printf("O106: varlen mrev on (nblk=%d, LPT expensive-first)\n", nblk_mv);
  } else if (mrev_flag) {
    printf("O106: varlen mrev requested but ignored (need causal & D==128/512 & nblk>=16)\n");
  }

  // O40：varlen LSE 的 K 维 split auto（D=128 目标 `grid*split≈2048`、cap 8；D=512 `≈256`、
  //   cap 16，与定长 O38/O39 同标定）；`--lsesplit=N`（>0）直接指定。再按最大序列的 tile 数封顶。
  const int nblk0 = (maxlen + LBM - 1) / LBM;
  // O58：把 fp16/bf16 的 O56（8-warp）/O57（2 CTA/SM）LSE 几何同构到 fp8。causal 默认走 cfg6
  //   （PIPE1/LBN16/NTH=128/LBM=64，smem 99.8KB → 2 CTA/SM）；full 保持 O54 旧默认，`--lseocc=5/6`
  //   或 `--lse8w=1` 可选（与 O56/O57 一致：full 上该杠杆为混合/负，未默认）。`--lse8w` 优先 8-warp。
  const int lse_nblk8 = (maxlen + 127) / 128;
  const bool lse8w_use = (lse8w != 0) && (D == 512);
  const bool causal_cfg6 = causal && (D == 512) &&
                           (lseocc == 5 || lseocc == 6 || (lseocc == 0 && !lse8w_use));
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    long base;
    if (D == 128 && lse_compact) base = (long)total_pairs * H;
    else if (D == 512 && !causal) base = (long)nblk0 * H * B;   // O54：full 一个 CTA 一个 m 块
    else base = (long)((nblk0 + 1) / 2) * H * B;
    // O58：causal 的 2 CTA/SM 几何（cfg5/6）tile 更细（fp8 LBN=16/HD=512 时 tile 更多），split
    //   目标提到 1024（b1→8 cap、b3→16，均为 A/B 最优附近）；旧默认/8-warp 维持 256。
    const int target = (D == 512) ? (causal ? (causal_cfg6 ? 1024 : 256) : 768) : 2048;
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
    // O64：默认把 4 次量化 + 3 次清零融合成 1 个 launch（`qfuseflag=0` 退回旧路径做 A/B）。
    // O66：`dfuseflag`（默认 1）时把 delta 也算进 dO 的量化任务（省一次 delta launch）。
    const bool dfuse = (dfuseflag != 0);
    if (qfuseflag) {
      const long long total = 3 * rq + 4 * rkv;
      const int g = (int)std::min<long long>((total + 3) / 4, 1048576);
      if (dfuse) {
        if (D == 128)
          quantize_zero_delta_warp_kernel<4><<<g, 128>>>(
              d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks,
              d_vs, d_dos, d_dq, d_dk, d_dv, rq, rkv);
        else
          quantize_zero_delta_warp_kernel<16><<<g, 128>>>(
              d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks,
              d_vs, d_dos, d_dq, d_dk, d_dv, rq, rkv);
      } else if (D == 128)
        quantize_zero_warp_kernel<4><<<g, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                 d_do8, d_qs, d_ks, d_vs, d_dos, d_dq, d_dk,
                                                 d_dv, rq, rkv);
      else
        quantize_zero_warp_kernel<16><<<g, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                  d_do8, d_qs, d_ks, d_vs, d_dos, d_dq, d_dk,
                                                  d_dv, rq, rkv);
    } else {
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
    }
    // LSE：grid.x 按 maxlen，逐 b 由 cu_seqlens 定界。
    //   causal → 镜像配对 + cp.async 的 wgmma 版（工作量随 mblk 递增，需均衡）；
    //   非 causal → 各 m 块工作量恒为 nblk 个 tile，本已均衡，走 O1 的 mma `lse_mma_kernel`。
    const int nblk = (maxlen + LBM - 1) / LBM;
    // O72（第 166 轮）：varlen full D=128 的 LSE 走 4D-TMA（对齐定长 O70）。TMA 描述符建在
    //   packed 张量上（dims={D,T,Hkv,1}，batch 维恒 0），内核对每个 b 用 `cu_seqlens[b]` 定界。
#if defined(FA_WGMMA) && defined(FA_TMA)
    const bool lse_tma_v = (g_lse_tma_varlen != 0) && g_lse_full_opt && (D == 128) && !causal;
    CUtensorMap qmap_v{}, kmap_v{};
    if (lse_tma_v) {
      qmap_v = make_lse_map_fp8(d_q8, H, T, D, 1);
      kmap_v = make_lse_map_fp8(d_k8, Hkv, T, D, 1);
    }
#endif
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
      else if (D == 512 && (lseocc == 5 || lseocc == 6 || causal_cfg6)) {
        // O58：causal MLA varlen 的 LSE 默认走 cfg6（2 CTA/SM，镜像配对版）；`--lseocc=5` 可选
        //   PIPE0/LBN32。grid 与默认相同（LBM=64 的镜像对）。
        if (lseocc == 5)
          launch_lse_bal<512, 0, false, 128, 32>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H,
                                                 Hkv, scale, d_cu, d_lse_part, lse_split_eff,
                                                 (long long)rows_q);
        else
          launch_lse_bal<512, 1, false, 128, 16>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H,
                                                 Hkv, scale, d_cu, d_lse_part, lse_split_eff,
                                                 (long long)rows_q);
      } else if (D == 512 && lse8w_use) {
        launch_lse_bal<512, 1, false, 256, 32>(dim3((lse_nblk8 + 1) / 2, H, B), d_q8, d_qs, d_k8,
                                               d_ks, d_lse, maxlen, H, Hkv, scale, d_cu,
                                               d_lse_part, lse_split_eff, (long long)rows_q);
      } else
        launch_lse_bal<512, 1>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, d_cu,
                               d_lse_part, lse_split_eff, (long long)rows_q);
    } else {
      dim3 lg(nblk, H, B);
      if (D == 128) {
        // O68（第 162 轮）：varlen full D=128 的 LSE 同定长，改走 O54 均衡版（cp.async 双缓冲）。
        // O72（第 166 轮）：再优先走定长 O70 同款的 4D-TMA（`--lsetmavarlen=0` 退回 cp.async 版）。
        bool did_tma_v = false;
#if defined(FA_WGMMA) && defined(FA_TMA)
        if (lse_tma_v) {
          launch_lse_bal_tma<128, 1, true>(lg, qmap_v, kmap_v, d_qs, d_ks, d_lse, maxlen, H, Hkv,
                                           scale, d_cu);
          did_tma_v = true;
        }
#endif
        if (!did_tma_v && g_lse_full_opt)
          launch_lse_bal<128, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                       scale, d_cu, nullptr, 1);
        else if (!did_tma_v)
          launch_lse<128>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, 0, d_cu);
      }
      else if (lseocc == 5 || lseocc == 6)
        // O58：full MLA varlen 的 LSE「2 CTA/SM」几何（O57 的 fp8 同构；默认仍是 O54 旧路）。
        if (lseocc == 5)
          launch_lse_bal<512, 0, true, 128, 32>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                                scale, d_cu, d_lse_part, lse_split_eff,
                                                (long long)rows_q);
        else
          launch_lse_bal<512, 1, true, 128, 16>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                                scale, d_cu, d_lse_part, lse_split_eff,
                                                (long long)rows_q);
      else if (lse8w_use)
        launch_lse_bal<512, 1, true, 256, 32>(dim3(lse_nblk8, H, B), d_q8, d_qs, d_k8, d_ks,
                                              d_lse, maxlen, H, Hkv, scale, d_cu, d_lse_part,
                                              lse_split_eff, (long long)rows_q);
      else if (lse_split_eff > 1)
        // O54：full MLA varlen 的 LSE 走 K 维 split（bal 的 FULL 版）。
        launch_lse_bal<512, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale,
                                     d_cu, d_lse_part, lse_split_eff, (long long)rows_q);
      else
        launch_lse<512>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, 0, d_cu);
    }
    const int d_rows = (int)rows_q;
    const int d_wpb = THREADS / 32;
    const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
    // O66：融合路径下 delta 已在 quant kernel 内算好，跳过独立的 delta launch。
    if (!(qfuseflag && dfuse)) {
      if (D == 128)
        delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
      else
        delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    }
    // 均衡分块：`--compact` 走紧凑一维 m-tile 网格（grid.z=1，由 mt_b/mt_m 查表解出 (b, mblk)）；
    //   默认走旧的 maxlen 网格（含早退死 CTA，实测反而更快，见 docs/03 §40）。
    dim3 mg = compact ? dim3(total_mt * ksplit, H, 1)
                      : dim3((maxlen + BM - 1) / BM * ksplit, H, B);
    // O106：非紧凑 varlen 网格把 O89 的 m 块反转表当作 `mt_m`（`mt_b=null` 时 b=blockIdx.z）。
    const int* mtb = compact ? d_mtb : nullptr;
    const int* mtm = compact ? d_mtm : d_mrev_v;
    if (D == 512) {
      // MLA：非 wgmma / 非 regdq（与定长 D=512 路径一致）。
      // O52：把定长路径的 O47（8-warp/256 线程）+ O51（K/V cp.async 回填流水）搬进 varlen MLA。
      //   -1=自动（默认 8-warp + kvpipe，与定长一致）；`--mla8w=0/1`、`--mlakvp=0/1` 强制 A/B。
      const bool w8 = (mla8w != 0);
      const bool kvp = w8 && (mla_kvp != 0);
      if (kvp)
        launch_bwd_main_kvpipe<512, 64, 32, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
      else if (w8)
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
      else
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
  // ---- O64 A/B：varlen 的融合 quant+zero 同 binary 端到端对比。----
  {
    auto time_all = [&](bool f) {
      qfuseflag = f ? 1 : 0;
      for (int i = 0; i < 3; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      return t / iters;
    };
    const float t_f = time_all(true), t_u = time_all(false);
    qfuseflag = 1;
    printf("[O64 A/B] VARLEN total fused(1 launch) %.4f ms | unfused(4 quant + 3 memset) %.4f ms "
           "(%.4fx)\n", t_f, t_u, t_u / t_f);
  }
  if (compact)
    printf("grid main = %d x %d x %d (compact, ksplit=%d, total_mt=%d, old-grid=%d) | use_regdq=%d | T=%d\n",
           total_mt * ksplit, H, 1, ksplit, total_mt, (maxlen + BM - 1) / BM * B, (int)use_regdq, T);
  else
    printf("grid main = %d x %d x %d (nocompact, ksplit=%d, total_mt=%d) | use_regdq=%d | T=%d\n",
           (maxlen + BM - 1) / BM * ksplit, H, B, ksplit, total_mt, (int)use_regdq, T);

  // ---- P3-4i A/B（D=128 varlen，`--det` / `--detk=N`）：把 P3-4e/f 的确定性 dK/dV/dQ 扩到
  //      变长。partial 布局沿用定长式、但 `S=maxlen`、`nblk=nblk_max`；归约改用
  //      `dkv_reduce_varlen_kernel`（按 `cu_seqlens` 的逐序列 `len_b/nblk_b` 定界、输出按
  //      packed token 定位）。同 session 计时 + 跑两遍 DET 验逐位可复现，再与 atomic 比
  //      max|diff|。强制 `REGDQ=true`（ksplit>1 时 dQ 走 partial 才能确定）。----
#ifdef FA_WGMMA
  constexpr bool kWgmVarlen = true;   // varlen D=128 构建恒为 `-DFA_WGMMA`，默认主 kernel 走 WGMMA
#else
  constexpr bool kWgmVarlen = false;
#endif
  if (det_ab && D == 128) {
    const int nblk_max = (maxlen + BM - 1) / BM;
    const int ks = det_ksplit < 1 ? 1 : det_ksplit;
    const size_t part_elems = (size_t)B * H * nblk_max * maxlen * D;   // 旧（maxlen-strided）上界
    // P3-4o：compact per-sequence offset（行前缀和，行 = 一个 (h,mblk,jg) 的 HD 向量）。
    //   每序列 b 只占 `H*nblk_b*len_b` 行 ⇒ 缩小地址跨度、提高二次归约 DRAM 局部性。
    std::vector<int> part_base_h(B + 1, 0);
    for (int b = 0; b < B; ++b) {
      const int len_b = cu[b + 1] - cu[b];
      const int nblk_b = (len_b + BM - 1) / BM;
      part_base_h[b + 1] = part_base_h[b] + H * nblk_b * len_b;
    }
    const size_t part_elems_cmp = (size_t)part_base_h[B] * D;
    int* d_part_base = nullptr;
    CUDA_CHECK(cudaMalloc(&d_part_base, (B + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_part_base, part_base_h.data(), (B + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    const int* pb = part_compact ? d_part_base : nullptr;   // 空 ⇒ 旧 maxlen-strided 布局（默认）
    const size_t dqpart_elems = (size_t)T * H * (size_t)ks * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    float* d_dq_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
    printf("[P3-4o] varlen D=128 partial: compact=%.1fMB (%.0f%% of old %.1fMB) layout=%s\n",
           part_elems_cmp * 4 / 1e6, 100.0 * (double)part_elems_cmp / (double)part_elems,
           part_elems * 4 / 1e6, pb ? "compact" : "maxlen-strided");
    auto run_at = [&](int k) {
      dim3 g(nblk_max * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      launch_bwd_main<128, 64, 32, true, kWgmVarlen, true, true, true>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, k, d_cu, nullptr, nullptr);
    };
    auto run_dt_p = [&](int k, const int* p) {
      dim3 g(nblk_max * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));   // ksplit==1 时 dQ 用无竞争 red_add2
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
      launch_bwd_main_det<128, 64, 32, true, true, true, true, THREADS, WN, kWgmVarlen>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, k, d_dk_part, d_dv_part, nblk_max, d_dq_part,
          d_cu, nullptr, nullptr, p);
      // P3-4n：dkv+dq 融合归约（默认）vs 两次 launch（`--nofusered`）。
      const int dkv_blocks = B * Hkv * maxlen;
      const int dq_blocks = (k > 1) ? T * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_varlen_kernel<128, 64><<<dkv_blocks + dq_blocks, 128>>>(
            d_dk_part, d_dv_part, (k > 1) ? d_dq_part : nullptr, d_dk, d_dv, d_dq, d_cu, H, Hkv,
            nblk_max, maxlen, (int)causal, k, dkv_blocks, p);
      } else {
        dim3 rg(B * Hkv, maxlen);
        dkv_reduce_varlen_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk, d_dv, d_cu, H,
                                                       Hkv, nblk_max, maxlen, (int)causal, p);
        if (k > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq, T, H, k);
        }
      }
    };
    auto run_dt = [&](int k) { run_dt_p(k, pb); };
    auto time_fn3 = [&](auto fn, float* out_ms) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_at = 0.f, ms_dt = 0.f;
    time_fn3([&] { run_at(ks); }, &ms_at);
    std::vector<float> ak(nkv), av(nkv), aq(nq);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    time_fn3([&] { run_dt(ks); }, &ms_dt);
    std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    run_dt(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    auto mad3 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[P3-4i A/B] varlen ksplit=%d | atomic %.4f ms | DET(partial+reduce) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
           ks, ms_at, ms_dt, ms_at / ms_dt, mad3(dq1, dq2), mad3(dk1, dk2), mad3(dv1, dv2),
           mad3(dq1, aq), mad3(dk1, ak), mad3(dv1, av));
    // ---- F4-b（第 134 轮）：把 F4 第一步的「fp16 partial + O62 写扇区化」从定长扩到 varlen。
    //      同 binary、只把 partial 存储从 fp32 换 fp16（`DET_HALF`）并让二次归约走 `P16`
    //      置换读（求和集合/次序不变）。两档 partial 应逐位可复现，数值只差 fp16 舍入。----
    __half* d_dk_part_h = nullptr;
    __half* d_dv_part_h = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part_h, part_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dv_part_h, part_elems * sizeof(__half)));
    auto run_dth_p = [&](int k, const int* p) {
      dim3 g(nblk_max * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<128, 64, 32, true, true, true, true, THREADS, WN, kWgmVarlen, false,
                          true>(g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                                d_lse, d_dq, d_dk, d_dv, maxlen, H, Hkv, scale, (int)causal, k,
                                reinterpret_cast<float*>(d_dk_part_h),
                                reinterpret_cast<float*>(d_dv_part_h), nblk_max, d_dq_part, d_cu,
                                nullptr, nullptr, p);
      const int dkv_blocks = B * Hkv * ((maxlen + 3) / 4);  // F4-c：P16 归约每 block 4 行
      const int dq_blocks = (k > 1) ? T * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_varlen_kernel<128, 64, true><<<dkv_blocks + dq_blocks, 128>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), (k > 1) ? d_dq_part : nullptr, d_dk,
            d_dv, d_dq, d_cu, H, Hkv, nblk_max, maxlen, (int)causal, k, dkv_blocks, p);
      } else {
        dim3 rg(B * Hkv, (maxlen + 3) / 4);  // F4-c：P16 归约每 block 4 行
        dkv_reduce_varlen_kernel<128, 64, true><<<rg, 128>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), d_dk, d_dv, d_cu, H, Hkv, nblk_max,
            maxlen, (int)causal, p);
        if (k > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq, T, H, k);
        }
      }
    };
    auto run_dth = [&](int k) { run_dth_p(k, pb); };
    float ms_dth = 0.f;
    time_fn3([&] { run_dth(ks); }, &ms_dth);
    std::vector<float> hdk1(nkv), hdv1(nkv), hdk2(nkv), hdv2(nkv);
    CUDA_CHECK(cudaMemcpy(hdk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    run_dth(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[F4 A/B] varlen ksplit=%d | DET-fp32 %.4f ms | DET-fp16(扇区化) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dk/dv=%.2e/%.2e | fp16-vs-fp32 dk/dv=%.2e/%.2e\n",
           ks, ms_dt, ms_dth, ms_dt / ms_dth, mad3(hdk1, hdk2), mad3(hdv1, hdv2),
           mad3(hdk1, dk1), mad3(hdv1, dv1));
    cudaFree(d_dk_part_h);
    cudaFree(d_dv_part_h);
    // ---- P3-4n A/B：varlen 二次归约 分离 vs 融合（只测 reduce；partial 已由上面 run_dt 预置）。
    {
      auto only_sep = [&]() {
        dim3 rg(B * Hkv, maxlen);
        dkv_reduce_varlen_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk, d_dv, d_cu, H,
                                                       Hkv, nblk_max, maxlen, (int)causal, pb);
        if (ks > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq, T, H, ks);
        }
      };
      auto only_fus = [&]() {
        const int dkv_blocks = B * Hkv * maxlen, dq_blocks = (ks > 1) ? T * H : 0;
        dkv_dq_reduce_varlen_kernel<128, 64><<<dkv_blocks + dq_blocks, 128>>>(
            d_dk_part, d_dv_part, (ks > 1) ? d_dq_part : nullptr, d_dk, d_dv, d_dq, d_cu, H, Hkv,
            nblk_max, maxlen, (int)causal, ks, dkv_blocks, pb);
      };
      float t_rs = 0.f, t_rf = 0.f;
      time_fn3(only_sep, &t_rs);
      std::vector<float> rsdk(nkv), rsdv(nkv), rsdq(nq);
      CUDA_CHECK(cudaMemcpy(rsdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rsdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rsdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      time_fn3(only_fus, &t_rf);
      std::vector<float> rfdk(nkv), rfdv(nkv), rfdq(nq);
      CUDA_CHECK(cudaMemcpy(rfdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rfdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rfdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4n A/B] varlen reduce separate(2 launch) %.4f ms | fused(1 launch) %.4f ms "
             "(%.3fx) | fused-vs-sep bitwise dq/dk/dv=%.2e/%.2e/%.2e\n",
             t_rs, t_rf, t_rs / t_rf, mad3(rfdq, rsdq), mad3(rfdk, rsdk), mad3(rfdv, rsdv));
    }
    // ---- P3-4o A/B：同 binary 只改 partial 布局（compact per-sequence vs maxlen-strided），
    //      主 kernel+二次归约一起计时；两布局数值应逐位相同（只改地址，不改求和集合/次序）。----
    {
      auto time_full = [&](const int* p, float* out_ms) {
        auto fn = [&]() { run_dt_p(ks, p); };
        for (int i = 0; i < 3; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        *out_ms = t / iters;
      };
      float t_cmp = 0.f, t_leg = 0.f;
      time_full(d_part_base, &t_cmp);
      std::vector<float> cdk(nkv), cdv(nkv), cdq(nq);
      CUDA_CHECK(cudaMemcpy(cdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      time_full(nullptr, &t_leg);
      std::vector<float> ldk(nkv), ldv(nkv), ldq(nq);
      CUDA_CHECK(cudaMemcpy(ldk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ldv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ldq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4o A/B] varlen ksplit=%d | maxlen-strided %.4f ms | compact %.4f ms (%.3fx) | "
             "compact-vs-legacy bitwise dq/dk/dv=%.2e/%.2e/%.2e\n",
             ks, t_leg, t_cmp, t_leg / t_cmp, mad3(cdq, ldq), mad3(cdk, ldk), mad3(cdv, ldv));
    }
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
    if (d_dq_part) cudaFree(d_dq_part);
    cudaFree(d_part_base);
    run_all();   // 恢复最终输出为 CLI 选中的路径
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- P3-4j/P3-4l A/B（D=512 MLA varlen，`--det`）：把 P3-4i 的确定性 dK/dV 从 D=128 扩到
  //      MLA（HD=512）的变长路径（DET 候选 ① 的最后一块）。dK/dV 走 partial +
  //      `dkv_reduce_varlen_kernel<512,64>`（按 `cu_seqlens` 的逐序列 `len_b/nblk_b` 定界、
  //      输出按 packed token 定位）。**P3-4l**：MLA 的 dQ 无法用寄存器累加
  //      （`kRegDq = REGDQ && (HD/NTW==1)` 恒 false），但 P3-4k 已让非 `kRegDq` 的 dQ epilogue
  //      在 `DET && ksplit>1` 时写按 part 分片的 `dq_part`（同 `(mblk,h,part)` 只被一个 CTA
  //      写 ⇒ 天然确定），故 ksplit>1 时再接 `dq_reduce_kernel<512>`（packed token = `qbase+qi`，
  //      与定长 `dq_reduce_kernel` 的 `row` 语义一致）按 part 固定次序求和；`ksplit==1` 仍走
  //      单写者 `red_add2`（逐位不变）。与默认 MLA varlen 主 kernel 同几何（O47/O51 的
  //      8-warp/256 线程），但用非 kvpipe 版（DET 未接进 kvpipe）。
  //      同 session 计时 + 跑两遍 DET 验逐位，再与 atomic 比 max|diff|。----
  if (det_ab && D == 512) {
    const int nblk_max = (maxlen + BM - 1) / BM;
    const int ks = det_ksplit < 1 ? 1 : det_ksplit;
    const size_t part_elems = (size_t)B * H * nblk_max * maxlen * D;   // 旧（maxlen-strided）上界
    // P3-4o：compact per-sequence offset（同 D=128 版；dK/dV 布局 HD 无关）。
    std::vector<int> part_base_h(B + 1, 0);
    for (int b = 0; b < B; ++b) {
      const int len_b = cu[b + 1] - cu[b];
      const int nblk_b = (len_b + BM - 1) / BM;
      part_base_h[b + 1] = part_base_h[b] + H * nblk_b * len_b;
    }
    const size_t part_elems_cmp = (size_t)part_base_h[B] * D;
    int* d_part_base = nullptr;
    CUDA_CHECK(cudaMalloc(&d_part_base, (B + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_part_base, part_base_h.data(), (B + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    const int* pb = part_compact ? d_part_base : nullptr;
    const size_t dqpart_elems = (size_t)T * H * (size_t)ks * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    float* d_dq_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
    printf("[P3-4o] varlen D=512 partial: compact=%.1fMB (%.0f%% of old %.1fMB) layout=%s\n",
           part_elems_cmp * 4 / 1e6, 100.0 * (double)part_elems_cmp / (double)part_elems,
           part_elems * 4 / 1e6, pb ? "compact" : "maxlen-strided");
    dim3 g(nblk_max * ks, H, B);
    auto run_at = [&]() {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, ks, d_cu, nullptr, nullptr);
    };
    // P3-4m：两个 DET 主 kernel 变体共用同一段 reduce（非 kvpipe 与 kvpipe 只差搬运，
    //   数值应逐位相同；kvpipe 是 O51 的 K/V cp.async 回填流水，此前 DET 未接线）。
    auto do_reduce = [&](const int* p) {
      const int dkv_blocks = B * Hkv * maxlen;
      const int dq_blocks = (ks > 1) ? T * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_varlen_kernel<512, 64><<<dkv_blocks + dq_blocks, 512>>>(
            d_dk_part, d_dv_part, (ks > 1) ? d_dq_part : nullptr, d_dk, d_dv, d_dq, d_cu, H, Hkv,
            nblk_max, maxlen, (int)causal, ks, dkv_blocks, p);
      } else {
        dim3 rg(B * Hkv, maxlen);
        dkv_reduce_varlen_kernel<512, 64><<<rg, 512>>>(d_dk_part, d_dv_part, d_dk, d_dv, d_cu, H,
                                                       Hkv, nblk_max, maxlen, (int)causal, p);
        if (ks > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<512><<<dg, 512>>>(d_dq_part, d_dq, T, H, ks);
        }
      }
    };
    auto run_dt_p = [&](const int* p) {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));   // ksplit==1 时 dQ 用无竞争 red_add2
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, ks, d_dk_part, d_dv_part, nblk_max, d_dq_part, d_cu,
          nullptr, nullptr, p);
      do_reduce(p);
    };
    auto run_dt = [&]() { run_dt_p(pb); };
    // P3-4m：DET + O51 K/V `cp.async` 回填流水（kvpipe）。
    auto run_dt_kv = [&]() {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4, false, true>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, ks, d_dk_part, d_dv_part, nblk_max, d_dq_part, d_cu,
          nullptr, nullptr, pb);
      do_reduce(pb);
    };
    auto time_fn3 = [&](auto fn, float* out_ms) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_at = 0.f, ms_dt = 0.f, ms_dt_kv = 0.f;
    time_fn3([&] { run_at(); }, &ms_at);
    std::vector<float> ak(nkv), av(nkv), aq(nq);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    time_fn3([&] { run_dt(); }, &ms_dt);
    std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    run_dt();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    // P3-4m：DET + kvpipe（数值应与非 kvpipe 逐位相同）。
    time_fn3([&] { run_dt_kv(); }, &ms_dt_kv);
    std::vector<float> dkk1(nkv), dvk1(nkv), dqk1(nq), dkk2(nkv), dvk2(nkv), dqk2(nq);
    CUDA_CHECK(cudaMemcpy(dkk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvk1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dqk1.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    run_dt_kv();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dkk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvk2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dqk2.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    auto mad3 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[P3-4l A/B] MLA varlen ksplit=%d | atomic(8w) %.4f ms | DET(8w) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
           ks, ms_at, ms_dt, ms_at / ms_dt, mad3(dq1, dq2), mad3(dk1, dk2), mad3(dv1, dv2),
           mad3(dq1, aq), mad3(dk1, ak), mad3(dv1, av));
    printf("[P3-4m A/B] MLA varlen ksplit=%d | DET nonkv %.4f ms | DET kvpipe %.4f ms (%.3fx) | "
           "kvpipe runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | kvpipe-vs-nonkv dq/dk/dv="
           "%.2e/%.2e/%.2e\n",
           ks, ms_dt, ms_dt_kv, ms_dt / ms_dt_kv, mad3(dqk1, dqk2), mad3(dkk1, dkk2),
           mad3(dvk1, dvk2), mad3(dqk1, dq1), mad3(dkk1, dk1), mad3(dvk1, dv1));
    // ---- F4-b（第 134 轮）：MLA（HD=512）varlen 的「fp16 partial + O62 扇区化」。
    //      `dkv_p16_perm` 只在 16 列块内置换、且 GEMM3/4 的 warp n-tile `c0=wc*GN34`
    //      （HD=512/8w：GN34=32）是 16 的倍数 ⇒ 全局列索引可直接套用，与定长/D=128 同一布局。----
    __half* d_dk_part_h = nullptr;
    __half* d_dv_part_h = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part_h, part_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dv_part_h, part_elems * sizeof(__half)));
    auto run_dth_p = [&](const int* p) {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4, false, false, true>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, ks, reinterpret_cast<float*>(d_dk_part_h),
          reinterpret_cast<float*>(d_dv_part_h), nblk_max, d_dq_part, d_cu, nullptr, nullptr, p);
      const int dkv_blocks = B * Hkv * ((maxlen + 3) / 4);  // F4-c：P16 归约每 block 4 行
      const int dq_blocks = (ks > 1) ? T * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_varlen_kernel<512, 64, true><<<dkv_blocks + dq_blocks, 512>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), (ks > 1) ? d_dq_part : nullptr, d_dk,
            d_dv, d_dq, d_cu, H, Hkv, nblk_max, maxlen, (int)causal, ks, dkv_blocks, p);
      } else {
        dim3 rg(B * Hkv, (maxlen + 3) / 4);  // F4-c：P16 归约每 block 4 行
        dkv_reduce_varlen_kernel<512, 64, true><<<rg, 512>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), d_dk, d_dv, d_cu, H, Hkv, nblk_max,
            maxlen, (int)causal, p);
        if (ks > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<512><<<dg, 512>>>(d_dq_part, d_dq, T, H, ks);
        }
      }
    };
    float ms_dth = 0.f;
    time_fn3([&] { run_dth_p(pb); }, &ms_dth);
    std::vector<float> hdk1(nkv), hdv1(nkv), hdk2(nkv), hdv2(nkv);
    CUDA_CHECK(cudaMemcpy(hdk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    run_dth_p(pb);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[F4 A/B] MLA varlen ksplit=%d | DET-fp32 %.4f ms | DET-fp16(扇区化) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dk/dv=%.2e/%.2e | fp16-vs-fp32 dk/dv=%.2e/%.2e\n",
           ks, ms_dt, ms_dth, ms_dt / ms_dth, mad3(hdk1, hdk2), mad3(hdv1, hdv2),
           mad3(hdk1, dk1), mad3(hdv1, dv1));
    cudaFree(d_dk_part_h);
    cudaFree(d_dv_part_h);
    // ---- P3-4o A/B：同 binary 只改 partial 布局（compact vs maxlen-strided）。
    {
      auto time_full = [&](const int* p, float* out_ms) {
        auto fn = [&]() { run_dt_p(p); };
        for (int i = 0; i < 3; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        *out_ms = t / iters;
      };
      float t_cmp = 0.f, t_leg = 0.f;
      time_full(d_part_base, &t_cmp);
      std::vector<float> cdk(nkv), cdv(nkv), cdq(nq);
      CUDA_CHECK(cudaMemcpy(cdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      time_full(nullptr, &t_leg);
      std::vector<float> ldk(nkv), ldv(nkv), ldq(nq);
      CUDA_CHECK(cudaMemcpy(ldk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ldv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ldq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4o A/B] MLA varlen ksplit=%d | maxlen-strided %.4f ms | compact %.4f ms (%.3fx) | "
             "compact-vs-legacy bitwise dq/dk/dv=%.2e/%.2e/%.2e\n",
             ks, t_leg, t_cmp, t_leg / t_cmp, mad3(cdq, ldq), mad3(cdk, ldk), mad3(cdv, ldv));
    }
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
    if (d_dq_part) cudaFree(d_dq_part);
    cudaFree(d_part_base);
    run_all();   // 恢复最终输出为 CLI 选中的路径
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- O52 A/B（D=512/MLA/varlen）：主 kernel 4-warp vs 8-warp(+kvpipe) 同 session 计时 ----
  //   只改 warp 网格与 K/V 搬运，数学口径不变；跨 CTA atomic 次序变 ⇒ 预期 max_abs ~1e-6。
  if (D == 512) {
    dim3 mgab = compact ? dim3(total_mt * ksplit, H, 1)
                        : dim3((maxlen + BM - 1) / BM * ksplit, H, B);
    const int* mtba = compact ? d_mtb : nullptr;
    const int* mtma = compact ? d_mtm : nullptr;
    auto go52 = [&](int mode) {  // 0=4w, 1=8w, 2=8w+kvpipe
      if (mode == 2)
        launch_bwd_main_kvpipe<512, 64, 32, false, true, true, true, 256, 4>(
            mgab, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtba, mtma);
      else if (mode == 1)
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mgab, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtba, mtma);
      else
        launch_bwd_main<512, 64, 32, false, false, true, true>(
            mgab, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtba, mtma);
    };
    auto zero_acc = [&]() {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
    };
    cudaEvent_t eva, evb;
    CUDA_CHECK(cudaEventCreate(&eva));
    CUDA_CHECK(cudaEventCreate(&evb));
    auto bench52 = [&](int mode, float* out) {
      zero_acc();
      go52(mode);
      CUDA_CHECK(cudaEventRecord(eva));
      for (int i = 0; i < iters; ++i) go52(mode);
      CUDA_CHECK(cudaEventRecord(evb));
      CUDA_CHECK(cudaEventSynchronize(evb));
      CUDA_CHECK(cudaEventElapsedTime(out, eva, evb));
      *out /= iters;
    };
    float m4 = 0.f, m8 = 0.f, mkv = 0.f;
    bench52(0, &m4);
    bench52(1, &m8);
    bench52(2, &mkv);
    auto grab52 = [&](int mode, std::vector<float>& dq, std::vector<float>& dk,
                      std::vector<float>& dv) {
      zero_acc();
      go52(mode);
      CUDA_CHECK(cudaMemcpy(dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    };
    std::vector<float> a4(nq), b4(nkv), c4(nkv), a8(nq), b8(nkv), c8(nkv), ak(nq), bk(nkv), ck(nkv);
    grab52(0, a4, b4, c4);
    grab52(1, a8, b8, c8);
    grab52(2, ak, bk, ck);
    auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
      double e = 0.0;
      for (size_t i = 0; i < x.size(); ++i) e = std::max(e, (double)fabs((double)x[i] - (double)y[i]));
      return e;
    };
    printf("[O52 A/B] varlen MLA main-only 4w %.4f ms | 8w %.4f ms (%.3fx) | 8w+kvpipe %.4f ms "
           "(%.3fx) | max_abs(8w-vs-4w) dq=%.3e dk=%.3e dv=%.3e | max_abs(kvp-vs-8w) "
           "dq=%.3e dk=%.3e dv=%.3e\n",
           m4, m8, m4 / m8, mkv, m4 / mkv, maxd(a4, a8), maxd(b4, b8), maxd(c4, c8),
           maxd(a8, ak), maxd(b8, bk), maxd(c8, ck));
    run_all();  // 恢复 CLI 选中路径（写回 d_dq/d_dk/d_dv）
  }

  // ---- O54 A/B（D=512/MLA/varlen/full）：非 causal LSE 的「bal FULL + K 维 split」vs 旧版。
  if (D == 512 && !causal) {
    const int nblk = (maxlen + LBM - 1) / LBM;
    auto go_lse = [&](int which, int sp) {
      if (which == 0)
        launch_lse<512>(dim3(nblk, H, B), d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, 0,
                        d_cu);
      else if (sp <= 1)
        launch_lse_bal<512, 1, true>(dim3(nblk, H, B), d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H,
                                     Hkv, scale, d_cu, nullptr, 1);
      else
        launch_lse_bal<512, 1, true>(dim3(nblk, H, B), d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H,
                                     Hkv, scale, d_cu, d_lse_part, sp, (long long)rows_q);
    };
    cudaEvent_t ela, elb;
    CUDA_CHECK(cudaEventCreate(&ela));
    CUDA_CHECK(cudaEventCreate(&elb));
    auto bench_lse = [&](int which, int sp, float* out) {
      go_lse(which, sp);
      CUDA_CHECK(cudaEventRecord(ela));
      for (int i = 0; i < iters; ++i) go_lse(which, sp);
      CUDA_CHECK(cudaEventRecord(elb));
      CUDA_CHECK(cudaEventSynchronize(elb));
      CUDA_CHECK(cudaEventElapsedTime(out, ela, elb));
      *out /= iters;
    };
    float t0 = 0.f;
    bench_lse(0, 1, &t0);
    printf("[O54 A/B] varlen MLA full LSE: old(O1 mma) %.4f ms |", t0);
    for (int sp : {1, 2, 4, 8, 16}) {
      float t = 0.f;
      bench_lse(1, sp, &t);
      printf(" balFULL/split%d %.4f", sp, t);
    }
    printf(" ms (auto=%d)\n", lse_split_eff);
    run_all();  // 恢复 CLI 选中路径
  }

  // ---- O58 A/B（D=512/MLA/varlen）：LSE 几何 sweep（LSE-only、同 session；causal 镜像配对 / full
  //   单 m 块）。默认 causal=cfg6、full=O54 旧路；8-warp 需 `--lse8w=1`。----
  if (D == 512) {
    dim3 lgc = causal ? dim3((nblk0 + 1) / 2, H, B) : dim3(nblk0, H, B);
    dim3 lg8c = causal ? dim3((lse_nblk8 + 1) / 2, H, B) : dim3(lse_nblk8, H, B);
    auto go_lse58 = [&](auto full_tag, int which, int sp) {
      constexpr bool F = decltype(full_tag)::value;
      if (which == 5)
        launch_lse_bal<512, 0, F, 128, 32>(lgc, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                           scale, d_cu, d_lse_part, sp, (long long)rows_q);
      else if (which == 6)
        launch_lse_bal<512, 1, F, 128, 16>(lgc, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                           scale, d_cu, d_lse_part, sp, (long long)rows_q);
      else if (which == 8)
        launch_lse_bal<512, 1, F, 256, 32>(lg8c, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                           scale, d_cu, d_lse_part, sp, (long long)rows_q);
      else
        launch_lse_bal<512, 1, F>(lgc, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale,
                                  d_cu, d_lse_part, sp, (long long)rows_q);
    };
    cudaEvent_t eca, ecb;
    CUDA_CHECK(cudaEventCreate(&eca));
    CUDA_CHECK(cudaEventCreate(&ecb));
    auto bench_lse_c = [&](auto full_tag, int which, int sp, float* out) {
      go_lse58(full_tag, which, sp);
      CUDA_CHECK(cudaEventRecord(eca));
      for (int i = 0; i < iters; ++i) go_lse58(full_tag, which, sp);
      CUDA_CHECK(cudaEventRecord(ecb));
      CUDA_CHECK(cudaEventSynchronize(ecb));
      CUDA_CHECK(cudaEventElapsedTime(out, eca, ecb));
      *out /= iters;
    };
    auto sweep58 = [&](auto full_tag, int which, const char* tag) {
      printf("[O58 A/B] %s:", tag);
      for (int sp : {1, 2, 4, 8, 16}) {
        float t = 0.f;
        bench_lse_c(full_tag, which, sp, &t);
        printf(" split%d %.4f", sp, t);
      }
      printf(" ms\n");
    };
    if (causal) {
      sweep58(std::integral_constant<bool, false>{}, 0, "legacy PIPE1/LBN64 4w (1 CTA)");
      sweep58(std::integral_constant<bool, false>{}, 6, "cfg6 PIPE1/LBN16 4w (2 CTA) [new default]");
      sweep58(std::integral_constant<bool, false>{}, 5, "cfg5 PIPE0/LBN32 4w (2 CTA)");
      sweep58(std::integral_constant<bool, false>{}, 8, "PIPE1/LBN32 8w (1 CTA)");
    } else {
      sweep58(std::integral_constant<bool, true>{}, 0, "legacy PIPE1/LBN64 4w (1 CTA) [default]");
      sweep58(std::integral_constant<bool, true>{}, 6, "cfg6 PIPE1/LBN16 4w (2 CTA)");
      sweep58(std::integral_constant<bool, true>{}, 5, "cfg5 PIPE0/LBN32 4w (2 CTA)");
      sweep58(std::integral_constant<bool, true>{}, 8, "PIPE1/LBN32 8w (1 CTA)");
    }
    run_all();  // 恢复 CLI 选中路径
  }

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
  // P3-3 varlen：`--dump=<prefix>` 把 packed 的 dq/dk/dv 落成 npy，供 fa_bwd_compare.py 汇总。
  if (!dump.empty()) {
    fa_bwd_save_npy_f32(dir + "/" + dump + "_dq.npy", h_dq.data(), (long long)h_dq.size());
    fa_bwd_save_npy_f32(dir + "/" + dump + "_dk.npy", h_dk.data(), (long long)h_dk.size());
    fa_bwd_save_npy_f32(dir + "/" + dump + "_dv.npy", h_dv.data(), (long long)h_dv.size());
    printf("[dump] ours -> %s/%s_{dq,dk,dv}.npy\n", dir.c_str(), dump.c_str());
  }
  return 0;
}

// =============================================================================
// host / launcher / self-test
// =============================================================================
int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8";
  std::string o_name = "ref_o";
  // P3-3：`--dump=<prefix>` 让 ours 的 dq/dk/dv 落成 `<prefix>_{dq,dk,dv}.npy`，
  //   供 harness/fa_bwd_compare.py 统一对拍（默认不落盘，行为逐位不变）。
  std::string dump_prefix;
  bool causal = true;
  int iters = 20;
  int ksplit = -1;  // -1 = 自动
  int ksplit2_opt = -1;  // O19：wg2 的 ksplit（-1 自动）
  // O89：定长 causal 主 kernel 的 m 块调度序（LPT）。0 = 历史（mblk = mt 升序，便宜块先跑）；
  //   1 = 反转（贵块先跑，削尾波）——只改「哪个 CTA 算哪个 m 块」，dK/dV 原子顺序略变，
  //   数值仍在 fp8 噪声内。用于验证「靠调度把 ksplit 降下来、消 Q/dO 重读」是否可行。
  int mrev_opt = 1;  // O89：默认开（LPT 贵块先跑）；--mrev=0 A/B
  // O93（第 188 轮，LPT 候选 ③）：定长 causal 默认 kernel 的**跨 head 全局 LPT**。0 = 历史
  //   （grid=(nblk*ksplit, H, B)，head 走慢轴 ⇒ 每个 head 内 m 降序的锯齿）；1 = 轴对调
  //   （grid=(H, nblk*ksplit, B)，head 走快轴）⇒ 所有 head 的最贵 m 块一起先派发。只改
  //   「哪个 CTA 算哪个 (h,mblk,part)」，dK/dV 原子顺序略变、数值在 fp8 噪声内。
  //   **默认开**（正结果）；`--hswap=0` A/B。配合 `hswap_elig` 时自动把 ksplit 收到 2
  //   （全局 LPT 让低 ksplit 的负载均衡够好，Q/dO 重读 8×→2×）。
  int hswap_opt = 1;  // O93：默认开；--hswap=0 回退历史锯齿序
  // O95（第 189 轮，O93 候选 ③）：**变 ks 调度**——`--ksm=N` 让最贵的 N 个 m 块用 ksplit=2、
  //   其余 m 块用 ksplit=1（表驱动、全局 LPT 序）。目的：在 O93 的 ksplit=2 平衡点上再砍掉
  //   便宜块那部分 Q/dO 重读（`l2 read`）与 dQ 跨-part 原子；只改「哪个 CTA 算哪段 K」，
  //   非 DET 默认路径的 dK/dV/dQ 仍是可交换跨 CTA `atomicAdd` ⇒ 数值在 fp8 噪声内。
  //   `-1` = 关（历史 O93 均匀 ksplit）。仅在 hswap eligible（定长 causal D=128 nblk>=16）生效。
  int ksm_opt = -1;  // O95：--ksm=N 开启（N=最贵 m 块数）；默认关
  int ksm_hi = 2;    // O95：最贵 N 个 m 块的 ks（默认 2）；`--ksmhi=K` A/B（如 3/4）
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
  int wg2wgmma = 0; // F6：1 = BM=128 双 warpgroup 主 kernel 的 GEMM1/2 走 wgmma（SW128）
  int wg2tma = 0;   // O90（F6-step3）：1 = wgmma2 的 K/V 改 4D-TMA（K 双缓冲）
  int wg3 = 0;        // O91（F6-step4）：1 = BM=192 / 3 warpgroup / 384 线程主 kernel
  int ksplit3_opt = -1;  // O91：wg3 的 ksplit（-1 自动）
  int prel_opt = -1;  // O12：-1 自动（开）；0/1 强制 LSE/D 预装寄存器开关
  int qfast = 1;      // O14：1 = warp-per-row 向量化量化，0 = 旧 per-row 标量量化（A/B）
  int delta_warp_opt = 1;  // O26：1 = warp-per-row 向量化 delta（默认），0 = 旧 per-row smem 归约（A/B）
  int f16b_opt = 1;   // O7e-2：1 = fold 16B 向量化写（默认），0 = 退回 O7e 的 4B 写（A/B）
  int bn64_opt = 0;   // O21：1 = 主 kernel KV tile BN=64（mma 路径，D=128）
  int cvt_on = 0;     // O21b：1 = 保留冗余的 fp32→fp32 convert 拷贝（默认 0：直接累加进输出）
  // O64：1 = 把 4 次输入量化 + 3 次累加缓冲清零融合成 1 个 launch（默认 1，数值逐位不变）；
  //   0 = 退回 O14 的 4 个 quant kernel + 3 个 cudaMemset，供同 session A/B。
  int qfuse = 1;
  // O88（本机 A/B）：融合 quant+zero kernel 的**栅格上限**（0 = 历史行为：grid = 任务数/4，
  //   即「每 warp 一行、每 CTA 4 行」，S4096 时 = 114688 个 CTA）。该 kernel 是纯带宽/访存
  //   kernel，114688 个微小 CTA 的**块调度开销**可能是端到端非 main 开销的一部分；把栅格
  //   封顶、让每 warp 沿 grid-stride 多搬几行可摊薄块调度。`--qcap=N` 强制上限（N=0 历史）。
  int qcap_opt = 0;
  // O66：1 = 把 delta 也融进 quant kernel 的 dO 任务（默认 1，数值逐位不变；需 qfuse=1
  //   且 delta 走 warp-per-row 版）；0 = 独立 delta launch，供同 session A/B。
  int dfuse = 1;
  int foldrcp_opt = 1;  // O27：1 = fold 量化用「每行 rcp + 乘法」（默认），0 = 精确除法（A/B）
  int regdq_opt = -1; // O22：-1 自动；0/1 强制关/开寄存器 dQ 累加（同 session A/B）
  // F6-③（O77）：TMA 描述符的 L2 promotion 档（0=NONE/1=L2_128B/2=L2_256B，仅 A/B）。
  int l2promo_opt = 0;
  // O32：LSE 是否用 TMA 版（仅 FA_TMA 构建、D==128、causal）。-1=自动（默认开），0/1 由
  //   `--lsetma=` 强制。
  int lse_tma = -1;
  // O37：主 kernel 的 Q/dO 是否用 4D-TMA（仅 FA_WGMMA+FA_TMA、D==128）。-1=自动（默认关，
  //   作 opt-in 与 cp.async 版同 binary A/B），0/1 由 `--qdtma=` 强制。
  int qd_tma = -1;
  // O41：主 kernel 的 K/V 是否也用 4D-TMA（roadmap「下一步候选 ①」）。仅 `-DFA_WGMMA -DFA_TMA`
  //   构建、D==128；-1=自动（默认开，对齐 O37），0/1 由 `--kvtma=` 强制。
  int kv_tma = -1;
  // O84（第 179 轮）：head_dim=256 的主 kernel 是否走 Hopper wgmma（GEMM1/2 换 wgmma、Q/dO/K/V
  //   存 SW128；非 TMA，走 cp.async 载入）。-1=自动（`-DFA_WGMMA` 构建默认开、否则退化 mma），
  //   0/1 由 `--d256wgm=` 强制（同 binary A/B）。数值与 mma 版同量级（只改 GEMM1/2 指令路径 +
  //   跨 CTA red 次序）。
  int d256wgm_opt = -1;
  // O85（第 180 轮）：head_dim=256 主 kernel 的 Q/dO 是否走 **4D-TMA**（chunk-major，NCH=2；
  //   K/V 仍 cp.async）。**A/B 中性 ⇒ 默认关**（`--d256tma=1` 开启，opt-in）。smem 与
  //   O84 的 cp.async wgmma 档相同（115,712B）⇒ 仍 2 CTA/SM。
  int d256tma_opt = -1;
  // O38：LSE 的 K 维 split 数（仅 D=128/causal/TMA 生效）。0=自动（目标 grid*split≈2048、上限 8），
  //   >=1 直接指定（`--lsesplit=N` 走「切片 partial + merge」；`--lsesplit=1` 退回 O32、保持历史逐位）。
  int lse_split = 0;
  // O47：MLA（D=512）主 kernel 的 warp 网格。-1=自动（D=512 默认开 8-warp/256 线程），
  //   0/1 由 `--mla8w=` 强制（同 session A/B）。默认 128/2 与历史逐字等价。
  int mla8w_opt = -1;
  // O51：MLA（D=512）主 kernel 的 K/V cp.async 回填流水。-1=自动（默认开），0/1 由 `--mlakvp=`
  //   强制（同 binary A/B；需 8-warp 几何）。
  int mla_kvp_opt = -1;
  // O58：MLA（D=512）varlen LSE 的几何开关。`--lseocc=5/6`（2 CTA/SM：PIPE0/LBN32 或
  //   PIPE1/LBN16）、`--lse8w=1`（8-warp/256 线程/LBM=128/LBN=32）；默认 causal 走 cfg6、
  //   full 走 O54 旧路。`--lseocc=4` 退回 causal 旧默认（PIPE1/LBN64）做 A/B。
  int lseocc_opt = 0;
  int lse8w_opt = 0;
  // O48（候选 ①）：D=128 主 kernel 是否也用 8-warp（256 线程 / 2×4 网格）几何。0=默认
  //   4-warp（128/2），1=8-warp。仅 mma 路径（WGMMA 版 GEMM1/2 是 warpgroup 级、结构上锁死
  //   2 warp）；用 `--d128w=0/1` 做同 binary A/B。见 docs/03 §48。O49：默认 **-1=自动**
  //   （仅 `!wgmma` 的 mma 路径、且含 ksplit 的有效 grid ≤ SM 数时开；fp8 的 auto split-K
  //   通常已把小 S 的 grid 抬到 ≫132，故自动档在默认 shape 下不触发、保持逐位）。
  int d128w_opt = -1;
  // O78（实验/诊断）：端到端 overlap 可行性探针。>0 时在默认计时后额外测「main 与 LSE
  //   并发（两条非阻塞 stream）」相对「串行」的墙钟，用来判定本卡能否靠重叠非 main 阶段
  //   提吞吐（ROADMAP「下一步候选 ②：preprocess/main 跨 head 流水」）。**只做计时诊断**，
  //   并发期的 LSE 输出被丢弃、不影响已 dump 的结果。`--ovltest=1`。
  int ovltest = 0;
  // O78（正结果候选）：端到端 overlap —— 把 q/dO/K/V 按 **head 分成 ovlp 块**，用两条非阻塞
  //   stream 把「LSE(chunk k+1)」与「main(chunk k)」重叠（quant 仍整体在流水前）。
  //   仅定长 / D=128 / causal / MHA(Hkv==H) / 默认 wgmma+4D-TMA / H%ovlp==0 时启用；
  //   否则逐字退回原串行路径。`--ovlp=N`（N≥2）。见 docs/03 §101。
  int ovlp = 0;
  int ovl_nolse = 0;   // O78 诊断控制：1=跳过 LSE 发射，只测「分块 main 串行」的下界。
  // P3-4e：1 = 跑「确定性 dK/dV（partial + 固定次序归约）」A/B（仅 D=128 定长 mma 默认路径，
  //   对齐 fp16/bf16 O7b；`--det` 或 `--det=1`）。默认关，不影响常规计时。
  int det_ab = 0;
  // P3-4f：DET 实验用的 ksplit。>1 时 dQ 也走 partial（确定性 + split-K 并行度）。
  //   `--detk=N`。**O102（第 196 轮）**：默认改为 0 = auto——定长 D=128 的 DET A/B 在
  //   causal 下取 k=4（标定触底）、full 取 k=1；`--detk=1` 可显式复现 P3-4e/g 原状。
  int det_ksplit = 0;
  // P3-4n：DET 的 dkv/dq 二次归约是否融合成一次 launch（默认 1）。`--nofusered` 退回两次
  //   launch 做同 binary A/B。两版求和次序逐字相同 ⇒ 数值逐位相同。
  int fuse_reduce = 1;
  // P3-4o：varlen DET 的 dK/dV partial 是否用 compact per-sequence 布局。默认 0 = 旧
  //   maxlen-strided（性能略优）。`--partcompact` 打开 compact（分配大幅缩小，但实测约
  //   0.966–0.975×，见 docs/03 §66）；A/B 在同 binary 内始终两种布局都跑。
  int part_compact = 0;
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
    else if (a.rfind("--lsefull=", 0) == 0) g_lse_full_opt = atoi(a.c_str() + 10);
    else if (a == "--lsefull") g_lse_full_opt = 1;
    else if (a.rfind("--lse512old=", 0) == 0) g_lse_full512_opt = !atoi(a.c_str() + 12);
    else if (a == "--lse512old") g_lse_full512_opt = 0;
    else if (a.rfind("--lsetmavarlen=", 0) == 0) g_lse_tma_varlen = atoi(a.c_str() + 15);
    else if (a == "--lsetmavarlen") g_lse_tma_varlen = 1;
    else if (a.rfind("--mla8w=", 0) == 0) mla8w_opt = atoi(a.c_str() + 8);
    else if (a.rfind("--mlakvp=", 0) == 0) mla_kvp_opt = atoi(a.c_str() + 9);
    else if (a == "--mlakvp") mla_kvp_opt = 1;
    else if (a.rfind("--lseocc=", 0) == 0) lseocc_opt = atoi(a.c_str() + 9);
    else if (a.rfind("--lse8w=", 0) == 0) lse8w_opt = atoi(a.c_str() + 8);
    else if (a == "--lse8w") lse8w_opt = 1;
    else if (a.rfind("--d128w=", 0) == 0) d128w_opt = atoi(a.c_str() + 8);
    else if (a.rfind("--ovltest=", 0) == 0) ovltest = atoi(a.c_str() + 10);
    else if (a.rfind("--ovlp=", 0) == 0) ovlp = atoi(a.c_str() + 7);
    else if (a.rfind("--ovlnolse=", 0) == 0) ovl_nolse = atoi(a.c_str() + 11);
    else if (a == "--d128w") d128w_opt = 1;
    else if (a.rfind("--det=", 0) == 0) det_ab = atoi(a.c_str() + 6);
    else if (a == "--det") det_ab = 1;
    else if (a.rfind("--detk=", 0) == 0) det_ksplit = atoi(a.c_str() + 7);
    else if (a == "--nofusered") fuse_reduce = 0;
    else if (a == "--partcompact") part_compact = 1;
    else if (a.rfind("--qdtma=", 0) == 0) qd_tma = atoi(a.c_str() + 8);
    else if (a.rfind("--kvtma=", 0) == 0) kv_tma = atoi(a.c_str() + 8);
    else if (a == "--kvtma") kv_tma = 1;
    else if (a.rfind("--d256wgm=", 0) == 0) d256wgm_opt = atoi(a.c_str() + 10);
    else if (a == "--d256wgm") d256wgm_opt = 1;
    else if (a.rfind("--d256tma=", 0) == 0) d256tma_opt = atoi(a.c_str() + 10);
    else if (a == "--d256tma") d256tma_opt = 1;
    else if (a == "--wg2") wg2 = 1;
    else if (a == "--wg2wgmma") wg2wgmma = 1;
    else if (a == "--wg2tma") wg2tma = 1;
    else if (a == "--wg3") wg3 = 1;
    else if (a.rfind("--ksplit3=", 0) == 0) ksplit3_opt = atoi(a.c_str() + 10);
    else if (a == "--bn64") bn64_opt = 1;
    else if (a.rfind("--cvt=", 0) == 0) cvt_on = atoi(a.c_str() + 6);
    else if (a.rfind("--qfuse=", 0) == 0) qfuse = atoi(a.c_str() + 8);
    else if (a.rfind("--dfuse=", 0) == 0) dfuse = atoi(a.c_str() + 8);
    else if (a.rfind("--qcap=", 0) == 0) qcap_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--foldrcp=", 0) == 0) foldrcp_opt = atoi(a.c_str() + 10);
    else if (a.rfind("--qfast=", 0) == 0) qfast = atoi(a.c_str() + 8);
    else if (a.rfind("--deltawarp=", 0) == 0) delta_warp_opt = atoi(a.c_str() + 12);
    else if (a.rfind("--regdq=", 0) == 0) regdq_opt = atoi(a.c_str() + 8);
    else if (a.rfind("--l2promo=", 0) == 0) l2promo_opt = atoi(a.c_str() + 10);
    else if (a.rfind("--prel=", 0) == 0) prel_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--f16b=", 0) == 0) f16b_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--dump=", 0) == 0) dump_prefix = a.substr(7);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--ksplit=", 0) == 0) ksplit = atoi(a.c_str() + 9);
    else if (a.rfind("--ksplit2=", 0) == 0) ksplit2_opt = atoi(a.c_str() + 10);
    else if (a.rfind("--mrev=", 0) == 0) mrev_opt = atoi(a.c_str() + 7);
    else if (a == "--mrev") mrev_opt = 1;
    else if (a.rfind("--hswap=", 0) == 0) hswap_opt = atoi(a.c_str() + 8);
    else if (a == "--hswap") hswap_opt = 1;
    else if (a.rfind("--ksm=", 0) == 0) ksm_opt = atoi(a.c_str() + 6);
    else if (a == "--ksm") ksm_opt = 0;
    else if (a.rfind("--ksmhi=", 0) == 0) ksm_hi = atoi(a.c_str() + 8);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  g_l2promo = l2promo_opt;  // F6-③：在创建 TMA 描述符之前生效

  if (varlen)
    return run_varlen(dir, causal, iters, compact_opt, lse_compact_opt, lse_split, mla8w_opt,
                       mla_kvp_opt, lseocc_opt, lse8w_opt, dump_prefix, det_ab, det_ksplit,
                       fuse_reduce, part_compact, qfuse, dfuse, ksplit, mrev_opt);

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
  if (D != 128 && D != 256 && D != 512) {
    fprintf(stderr, "本版本支持 head_dim=128（MHA/GQA）/ 256 / 512（MLA）；当前 %d\n", D);
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
  // O76（第 171 轮）：新增 head_dim=256（D=128 与 D=512 之间，FA/TE 反向支持到 256）；
  //   走 mma 主 kernel（无 wgmma/TMA——fp8 wgmma 只做 HD=128），smem ≈115KB ⇒ 1 CTA/SM。
  const int smem_bytes = (D == 128)   ? Fp8Cfg<128, 64, 32>::smem_bytes
                         : (D == 256) ? Fp8Cfg<256, 64, 32>::smem_bytes
                                      : Fp8Cfg<512, 64, 32>::smem_bytes;
  const int lse_smem = (D == 128)   ? Fp8Cfg<128, 64, 32>::lse_smem_bytes
                       : (D == 256) ? Fp8Cfg<256, 64, 32>::lse_smem_bytes
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

  // O89：定长 causal 的 LPT m 块调度序（`--mrev=1`）。构建反转的 mt→mblk 查询表，透传给
  //   默认 kvtma 主 kernel（稠密网格 grid.x = nblk*ksplit，mt = blockIdx.x/ksplit）。
  //   只改「哪个 CTA 算哪个 m 块」；dK/dV 的跨 CTA 原子顺序略变 ⇒ 数值在 fp8 噪声内。
  int* d_mrev = nullptr;
  // O104（第 198 轮）：把 O93 的跨 head 全局 LPT（hswap）从 D=128 推广到 **D=256 的通用 wgmma
  //   主 kernel**。实测（本卡，见 docs/03 §126）：hswap 对 D=256 的收益**只在小 `base_grid`
  //   （=`nblk*H*B ≤ 256`，即 K/V 工作集小、hswap 损 L2 局部性可忽略）时为正**——
  //   S1024H8 1.15×、S1024H16kv4 1.09×、S2048H8 1.02×；而 S2048H16（base=512）、S4096H8
  //   （base=512）为负（0.95×/0.84×）。故只在 `base_grid ≤ 256` 且 causal 定长 nblk≥16 时启用。
  // O105（第 199 轮）：把 O89 的 **per-head LPT m 块反转**（`mblk = nblk-1-mt`，grid 与 head
  //   排布都不变）从 D=128 推广到 **D=512（MLA）causal 定长**。对照实验：D=256 大 `base_grid`
  //   （S2048H16/S4096H8）的 mrev 仅 1.003×（噪声内），故**不纳入 D=256 默认**（其小 base_grid
  //   档已由 O104 的 hswap256 覆盖）；D=512 的 S1024H2（nblk=16，1 CTA/SM、`target=S/2` 切得细）
  //   实测 main **1.068×**（反复重测稳定），是 O104 之后 MLA 主 kernel 的第一条调度正收益。
  //   只改「哪个 CTA 算哪个 m 块」，dK/dV 的跨 CTA 原子次序略变 ⇒ 数值仍在 fp8 噪声内。
  //   `--mrev=0` 回退历史（mt 升序、便宜块先跑）做同 binary A/B。
  const bool mrev_elig = (D == 128) || (D == 512) ||
                         (D == 256 && hswap_opt != 0 &&
                          (long)((S + 63) / 64) * H * B <= 256);
  if (mrev_opt && causal && mrev_elig && (S + 63) / 64 >= 16) {  // O89：nblk>=16（S>=1024）才启用，避免小 S 噪声
    const int nblk_m = (S + 63) / 64;
    std::vector<int> hrev(nblk_m);
    for (int i = 0; i < nblk_m; ++i) hrev[i] = nblk_m - 1 - i;
    CUDA_CHECK(cudaMalloc(&d_mrev, nblk_m * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mrev, hrev.data(), nblk_m * sizeof(int), cudaMemcpyHostToDevice));
    printf("O89: mrev on (nblk=%d, LPT expensive-first)\n", nblk_m);
  } else if (mrev_opt) {
    printf("O89/O105: mrev requested but ignored (need causal & D==128/256/512 & fixed-length & nblk>=16)\n");
  }
  // O93：跨 head 全局 LPT（轴对调）的启用条件（与 launch 分支一致）：默认开 + mrev 表已建
  //   （定长 causal D=128 nblk>=16）。仅 Hopper `-DFA_WGMMA -DFA_TMA` 构建真正生效。
  const bool hswap_elig = (hswap_opt != 0) && (mrev_opt != 0) && causal && D == 128 &&
                          (S + 63) / 64 >= 16;
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (hswap_elig)
    printf("O93: hswap on (grid=(H,nblk*ksplit,B), global LPT across heads)\n");
  else if (hswap_opt && D == 128)
    printf("O93: hswap requested but ignored (need causal & D==128 & nblk>=16)\n");
#else
  if (hswap_opt && D == 128) printf("O93: hswap requested but ignored (非 Hopper 构建)\n");
#endif
  // O104（第 198 轮）：把 O93 的跨 head 全局 LPT 从 `kvtma`（D=128）推广到**通用 wgmma 主 kernel**
  //   （D=256 默认档）。只需 `-DFA_WGMMA`（不需 TMA——D=256 走 cp.async）；`--hswap=0` 关闭。
  const bool hswap256_elig = (hswap_opt != 0) && (mrev_opt != 0) && causal && D == 256 &&
                             (S + 63) / 64 >= 16 &&
                             (long)((S + 63) / 64) * H * B <= 256;
#if defined(FA_WGMMA)
  if (hswap256_elig)
    printf("O104: hswap256 on (D=256, grid=(H,nblk*ksplit,B), global LPT across heads)\n");
  else if (hswap_opt && D == 256 && (long)((S + 63) / 64) * H * B > 256)
    printf("O104: hswap256 skipped for D=256 (base_grid=%ld > 256: L2 locality loss > LPT gain)\n",
           (long)((S + 63) / 64) * H * B);
#else
  if (hswap_opt && D == 256) printf("O104: hswap256 requested but ignored (非 Hopper 构建)\n");
#endif
  // O95：变 ks 调度表（`--ksm=N`）。仅 hswap eligible 时构建；表按 mt 升序（= 全局 LPT）排列，
  //   前 N 个（最贵）m 块 ks=2、其余 ks=1。`slot_tab` 由主 kernel 在 HSWAP 路径消费。
  int* d_ksm = nullptr;
  int ksm_nslots = 0;
  if (hswap_elig && ksm_opt >= 0) {
    const int nblk_m = (S + 63) / 64;
    int n2 = ksm_opt;
    if (n2 > nblk_m) n2 = nblk_m;
    if (n2 < 0) n2 = 0;
    std::vector<int> tab;
    tab.reserve((size_t)nblk_m + n2);
    const int ks_hi = (ksm_hi < 1) ? 1 : ((ksm_hi > 15) ? 15 : ksm_hi);
    for (int mt = 0; mt < nblk_m; ++mt) {
      const int ks_i = (mt < n2) ? ks_hi : 1;
      for (int part = 0; part < ks_i; ++part)
        tab.push_back((mt << 8) | (part << 4) | ks_i);
    }
    ksm_nslots = (int)tab.size();
    CUDA_CHECK(cudaMalloc(&d_ksm, ksm_nslots * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ksm, tab.data(), ksm_nslots * sizeof(int), cudaMemcpyHostToDevice));
    printf("O95: ksm on (top %d/%d m-blocks ks=%d, rest ks=1; nslots=%d vs uniform ks=2 %d)\n", n2,
           nblk_m, ks_hi, ksm_nslots, 2 * nblk_m);
  } else if (ksm_opt >= 0) {
    printf("O95: ksm requested but ignored (need hswap-eligible: causal & D==128 & nblk>=16)\n");
  }

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
    } else if (D == 256) {
      quantize_row_warp_kernel<8, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<8, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<8, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<8, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    } else {
      quantize_row_warp_kernel<16, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<16, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    }
  };
  auto quant = [&]() { if (qfast) quant_new(); else quant_old(); };
  // O64：融合「4 次量化 + 3 次清零」的单一 launch（数值与 quant()+3×memset 逐位相同）。
  auto quant_zero = [&]() {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const long long total = 3 * rq + 4 * rkv;
    int grid = (int)std::min<long long>((total + 3) / 4, 1048576);
    if (qcap_opt > 0) grid = std::min(grid, qcap_opt);  // O88：栅格封顶（A/B）
    if (D == 128)
      quantize_zero_warp_kernel<4><<<grid, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                  d_do8, d_qs, d_ks, d_vs, d_dos, d_dq_acc,
                                                  d_dk_acc, d_dv_acc, rq, rkv);
    else if (D == 256)
      quantize_zero_warp_kernel<8><<<grid, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                  d_do8, d_qs, d_ks, d_vs, d_dos, d_dq_acc,
                                                  d_dk_acc, d_dv_acc, rq, rkv);
    else
      quantize_zero_warp_kernel<16><<<grid, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                   d_do8, d_qs, d_ks, d_vs, d_dos, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, rq, rkv);
  };
  // O66：在 O64 融合的基础上，把 delta 也算进 dO 的量化任务（省一次独立 delta launch）。
  auto quant_zero_delta = [&]() {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const long long total = 3 * rq + 4 * rkv;
    int grid = (int)std::min<long long>((total + 3) / 4, 1048576);
    if (qcap_opt > 0) grid = std::min(grid, qcap_opt);  // O88：栅格封顶（A/B）
    if (D == 128)
      quantize_zero_delta_warp_kernel<4><<<grid, 128>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv);
    else if (D == 256)
      quantize_zero_delta_warp_kernel<8><<<grid, 128>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv);
    else
      quantize_zero_delta_warp_kernel<16><<<grid, 128>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv);
  };

  // ---- O2b：自动选择 N 方向切块数 ksplit。base = 未切块时的 CTA 数；切块把小 S 时
  //      不足一个波、或大 S 的尾波（partial wave）用更细的 CTA 补满并发槽。
  //      实测（docs/03 §15 的 ksplit sweep）：d128（smem 73.8KB→3 CTA/SM）在
  //      grid≈4096（≈10 个波）时最优；MLA d512（smem 223KB→1 CTA/SM）在 grid≈一个波
  //      （132）时最优——再切只增 prologue 与 dQ atomic 竞争。故按 head_dim 取目标：
  //        TARGET = (D==128) ? 4096 : 132;  k = clamp(TARGET/base, 1, 16) 后向下取 2 的幂。----
  constexpr int BM = 64;
  const long base_grid = (long)((S + BM - 1) / BM) * H * B;
  const bool ksplit_auto = (ksplit < 1);
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
  // O96（第 190 轮）：**full（非 causal）D=128 的 ksplit 重标定**。O29 的 target（2048/8192）
  //   是按 **causal 三角偏斜**标定的——causal 下 m 块工作量 ∝(m+1)，要靠极细切分把尾波摊平；
  //   而 **full 每块工作量相同**，尾波只由「grid 是否落在整数个并发波上」决定，过细切分只剩下
  //   Q/dO 重读 + dQ 跨 part 原子（纯浪费）。实测（S512/1024/2048 H8/16/32，见 docs/03 §118）：
  //   自动档 k=8/16 比最优慢 **1.05–1.59×**；最优 k 一致地让 `grid=base*k` 最接近整数个并发波
  //   （本卡 D=128 fp8 main = 3 CTA/SM × 132 SM = **396 槽**）。故 full D=128 自动档改为：在
  //   k∈[1,8] 里取「尾波空泡 `ceil(g/SLOTS)*SLOTS - g`」最小者，并列取更小 k（Q/dO 重读更少）。
  //   显式 `--ksplit=K` 时不覆盖。
  if (ksplit_auto && !causal && D == 128) {
    const long SLOTS = 396L;  // 3 CTA/SM × 132 SM
    long best_k = 1, best_waste = -1;
    for (long k = 1; k <= 8; ++k) {
      const long g = base_grid * k;
      const long waste = ((g + SLOTS - 1) / SLOTS) * SLOTS - g;
      if (best_waste < 0 || waste < best_waste) {
        best_waste = waste;
        best_k = k;
      }
    }
    ksplit = (int)best_k;
  }
  // O97（第 191 轮）：**full（非 causal）D=256 / D=512 的 ksplit 重标定**（O96 的同类审计推广）。
  //   D=256 与 D=512（MLA）共用 `target_ctas = S/2`（O29 按 **causal** MLA 标定）。对两者的
  //   **full** 这都不对：full 每块工作量相同，最优 k 由「并发槽利用率」而非「三角尾波」决定。
  //   · **D=512（MLA，1 CTA/SM → 132 槽）**：target=S/2 在 base 小时把 k 顶到 16，实测过切——
  //     5 个 full shape 的最优是 k≈128/base（S512H2 k8、S1024H2/S512H4 k4、S2048H2 k2），
  //     正是「按 132 槽做波对齐」（main 1.14–1.19×）。大 S（S4096H2）波对齐给 k=1 略欠（1.7%），
  //     但远好于 auto 的 k=16。
  //   · **D=256（2 CTA/SM → 264 槽）**：`S/2 / base = 32/H` 与 S 无关 ⇒ 大 S 严重欠切——7 个
  //     shape 的最优 k 稳定在 **8–12**（auto 只 1–4，main 慢 **5.7–8.3%**）；S<2048 则按 264 槽
  //     波对齐即可（S1024H16 甚至 k=1 最优）。
  //   显式 `--ksplit=K` 时不覆盖（`ksplit_auto` 已判定）。
  if (ksplit_auto && !causal && (D == 256 || D == 512)) {
    const bool wave = (D == 512) || (S < 2048);
    if (wave) {
      const long SLOTS = (D == 512) ? 132L : 264L;  // 1 / 2 CTA/SM × 132 SM
      const long KMAX = (D == 512) ? 16 : 8;
      long best_k = 1, best_waste = -1;
      for (long k = 1; k <= KMAX; ++k) {
        const long g = base_grid * k;
        const long waste = ((g + SLOTS - 1) / SLOTS) * SLOTS - g;
        if (best_waste < 0 || waste < best_waste) { best_waste = waste; best_k = k; }
      }
      // D=512 的 base 已 ≥ 一整个波（S4096H2 波对齐给 k=1）时，长 K 循环仍偏好 ≥2 份并发
      //   （实测 k=1 2.491ms vs k=2 2.474ms vs 旧 auto k=16 2.473ms）⇒ 给 k 设下限 2 消除回退。
      if (D == 512 && best_k < 2) best_k = 2;
      ksplit = (int)best_k;
    } else {  // D==256 && S>=2048：给足并发（grid≈8192，≈31 波），cap k=12
      long k = 8192L / base_grid;
      if (k < 1) k = 1;
      if (k > 12) k = 12;
      ksplit = (int)k;
    }
  }
  // O99（第 193 轮）：**causal D=256 的 ksplit 重标定**。`D != 128` 一直沿用 O29 的
  //   `target_ctas = S/2`——那是按 **causal MLA（D=512，1 CTA/SM）** 标的，套到 **causal
  //   D=256（2 CTA/SM→264 槽）** 上 `k = S/2 / ((S/64)*H*B) = 32/(H*B)` 与 S 无关，小/中 S
  //   严重欠切（并发不足、藏不住延迟）。O96/O97/O98 只复核了 **full**，causal D=256 从未审计。
  //   实测（6 个 causal D=256 shape 全扫 k∈[1,32]，见 docs/03 §121）：
  //   · s512H8 最优 k=8、s1024H8 k=8/16（差 1.7%）、s2048H8 k=16、s4096H8 k=16、
  //     s1024H16kv4 k=6（k=8 差 0.5%）、s2048H16 k=12（k=8 差 0.6%）。
  //   · 最优都落在「`grid = base*k ≈ 2*S`」附近（即 `k ≈ 128/(H*B)`）：s1024→2048、s2048→4096、
  //     s4096→8192（H8 取到 cap 16）；小 S 被 `k ≤ nblk` 封顶（s512 取 8）。这与 D=128 causal
  //     大 S 的目标 `8192 = 2*S`（O29）同源。
  //   ⇒ 规则：`k = clamp(2*S/base, 1, 16)` 再按 `nblk` 封顶；`--ksplit=K` 显式时不覆盖。
  //   6 shape 全部 ≥1.05×（s1024H16 1.14×）、无回退。**只改「哪些 CTA 算哪段 K」⇒ 只改跨 CTA
  //   atomicAdd 次序，数值逐位不变。** full D=256/512 走 O97 分支、D=128 走 O96/O93，均不受影响。
  if (ksplit_auto && causal && D == 256) {
    const long nblk = (long)((S + 63) / 64);
    long k = (2L * S) / base_grid;
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    if (k > nblk) k = nblk;
    ksplit = (int)k;
  }
  // O93：hswap（跨 head 全局 LPT）启用时把自动 ksplit 收到 2——全局 LPT 使低 ksplit 的负载
  //   均衡足够好（S4096 main 1.443→1.373ms，Q/dO 重读 8×→2×；S1024H32/GQA 同向 1.10–1.12×）。
  //   仅默认 Hopper kvtma 构建生效；`--ksplit=K` 显式给出时不覆盖。（`hswap_elig` 定义见上。）
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (hswap_elig && ksplit_auto) ksplit = 2;
#endif
  // O104：D=256 hswap 把自动 ksplit 收到 **4**（实测最优；k=2 过细、并行度不足，k=8/16 又损
  //   Q/dO 重读 + 局部性）。`--ksplit=K` 显式给出时不覆盖。
#if defined(FA_WGMMA)
  if (hswap256_elig && ksplit_auto) ksplit = 4;
#endif
  // O7：只有 HD=128（dQ 一次铺满 N）且「平均每 CTA 的 nt tile 足够多」时才启用寄存器累加。
  // 因果下每 mblk 的 nt tile 数 ≈ (m0+BM)/BN，三角求和 /(mblk·ksplit) 后平均每 CTA
  // ≈ (S/BN)/2/ksplit；阈值取 4（实测 S=1024H32 平均=2、启用反而持平/略慢，S=4096=16 明显收益）。
  // O96：**full 下无三角折半**，平均每 CTA 的 nt tile 数 = (S/BN)/ksplit，故阈值判断不得再 `/2`
  //   （否则 full 在 k 偏大时被误关 regdq ⇒ dQ 逐 tile 跨 CTA `red`，S1024H16 从 0.22ms 退化到
  //   0.30ms）。causal 分支逐字不变。
  bool use_regdq = (D == 128) && ((long)(S / 32) / (causal ? 2 : 1) / ksplit >= 4);
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
  // O91（F6-step4）：wg3（BM=192、3 warpgroup、384 线程、1 CTA/SM）的 grid。并发槽 132，
  //   目标 ~8 波 ⇒ ksplit3 自动（base=(S/192)*H*B，S4096H16=352 时 k=8）；--ksplit3= 可覆盖。
  int ksplit3 = 4;
  if (ksplit3_opt >= 1) ksplit3 = ksplit3_opt;
  else {
    const long base_grid3 = (long)((S + 191) / 192) * H * B;
    long k = 4096L / (base_grid3 > 0 ? base_grid3 : 1);
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    long kp = 1;
    while (kp * 2 <= k) kp *= 2;
    ksplit3 = (int)kp;
  }
  dim3 mg3((S + 191) / 192 * ksplit3, H, B);
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
  // O74（第 169 轮）：MLA（D=512）的 causal LSE 也上 4D-TMA（此前只有 mma/cp.async 版）。
  if (D == 512) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<512, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Fp8Cfg<512, 64, 32>::lse_smem_bytes_tma1));
  }
  // O103（第 197 轮）：head_dim=256 的 LSE 也上 4D-TMA。LSE TMA kernel 在 O74 已泛化为
  //   `NCH=HD/128` 个 box（HD=256 → NCH=2），此前只是 host 未把它接到 D=256（`不做 4D-TMA
  //   （只服务 128/512）`）。host 建 D=256 的 Q/K 描述符 + 在此设 `lse_smem_bytes_tma1`。
  if (D == 256) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<256, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Fp8Cfg<256, 64, 32>::lse_smem_bytes_tma1));
  }
  if (lse_tma < 0) lse_tma = (D == 128 || D == 256) ? 1 : (D == 512 && causal ? 1 : 0);
#else
  if (lse_tma < 0) lse_tma = 0;
#endif
  printf("O32: lse backend = %s\n", lse_tma ? "tma" : "wgmma/mma");
  // O38：自动 split 档（0=auto）。目标 `grid*split ≈ 2048`（≈2 个满波；实测该目标在各 shape
  //   上距 per-shape 最优 ≤1.2%），上限 8；grid 已够大则退回 1（逐位）。
  // O59：把 O58 的 causal MLA LSE 几何（cfg6 默认，2 CTA/SM）从 varlen 推广到**定长**路径。
  //   O58 只改了 `run_varlen`，定长 MLA 的 causal LSE 一直用旧默认 `<512,1>`（PIPE1/LBN64）。
  //   以 `--lseocc=4` 退回旧默认、5=PIPE0/LBN32、6=PIPE1/LBN16；`--lse8w=1` 在定长不启用。
  const bool lse8w_fixed = (lse8w_opt != 0) && (D == 512);
  const bool causal_cfg6_fixed =
      causal && (D == 512) &&
      (lseocc_opt == 5 || lseocc_opt == 6 || (lseocc_opt == 0 && !lse8w_fixed));
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    // O101（第 195 轮）：base 随 causal/full 取不同网格——causal 走镜像配对（`lg_bal.x`），
    //   full 一个 CTA 一个 m 块（`lg.x=nblk`）。此前 full 也误用 `lg_bal.x`（只有 causal 的半格），
    //   对新增了 split 的 full D=256/D=512 会把并发目标低估一半。
    long lg_grid = (long)(causal ? lg_bal.x : lg.x) * H * B;
    // O39：D=512（MLA，mma LSE）目标 `grid*split ≈ 256`、上限 16；D=128 的 TMA LSE 维持
    //   O38 的 `≈2048`、上限 8。O59：cfg6 的 4 CTA/SM 把并发槽翻倍，目标抬到 1024
    //   （对齐 O58 varlen 的 fp8 档）。
    // O63（第 138 轮）：D=128 的 TMA LSE 在大 S 上原目标 2048 会比实测最优**多切一档**——
    //   S=4096 H16 时 base=`lg_grid`=512，2048→split=4，而同 binary 交替实测 split=2 的
    //   LSE 快 ~2.7%（端到端在噪声内）。S≥2048 时把目标降到 1024（仍 ≈2 个满波量级）；
    //   S<2048 维持 2048，避免小 shape 的二次归约/尾部回归（S512/S1024H32 实测两档相同）。
    // O101：full D=512 新接 split，目标取 **256**（同 causal legacy；实测 5 个 full MLA shape
    //   在 `base*split≈256` 时最优——S512H2/H4、S1024H2 取 8，S2048H2 取 4，S4096H2 取 2，
    //   与 `lg.x=nblk` 的 base 一致）。full D=256 新接 split，目标取 **512**（8 个 full D=256
    //   shape 实测：S512H8 取 4–8、S1024H8/H16 取 2–4、S2048H8 取 4，大 S 退化为 1）。
    //   causal 路径（D=512 cfg6/legacy、D=256）逐字不变。
    const int target = (D == 512) ? (causal ? (causal_cfg6_fixed ? 1024 : 256) : 256)
                                  : (causal ? (S >= 2048 ? 1024 : 2048) : 512);
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
  if (D == 128 || D == 256 || D == 512) {  // O74/O103：MLA（512）causal、D=256 LSE 需 Q/K 描述符
    qmap_lse = make_lse_map_fp8(d_q8, H, S, D, B);
    kmap_lse = make_lse_map_fp8(d_k8, Hkv, S, D, B);
  }
  // O37：主 kernel 的 Q/dO TMA 描述符（box={128,BM=64}，与 LSE 同一 dims={D,S,H,B}）。
  if (qd_tma < 0) qd_tma = 1;   // 默认开（对齐 O32 的 lsetma；`--qdtma=0` 供 A/B）
  // O41：K/V TMA 默认开（依赖 Q/dO TMA；`--kvtma=0` 供 A/B）。
  if (kv_tma < 0) kv_tma = 1;
  CUtensorMap qmap_main, dmap_main;
  // 注：O37 A/B 段无论 `--qdtma` 取值都会跑 TMA 版，故 D==128/256 时始终建描述符
  //     （否则 `--qdtma=0` 会拿未初始化 map 启动 TMA kernel → illegal instruction）。
  // O85：D=256 的 Q/dO 也走 4D-TMA（chunk-major，NCH=2），同样需要描述符。
  if (D == 128 || D == 256) {
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

  // O66：`do_delta=false` 时跳过 delta（融合路径已在 quant kernel 内算好）。
  auto run_preprocess = [&](bool do_delta = true) {
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
      } else if (g_lse_full_opt) {
        // O68（第 162 轮）：非 causal D=128 的 LSE 从 O1 `lse_mma_kernel`（无 cp.async 流水）
        //   改走 O54 的均衡版 `lse_mma_kernel_bal<FULL=true>`——一个 CTA 一个 m 块、
        //   K 用 `cp.async` 16B 双缓冲。full 各 m 块工作量相同（均 nblk 个 tile）无需镜像配对。
        // O70（第 164 轮）：full D=128 定长**优先走 4D-TMA**（对齐 causal O32 的搬运方式，
        //   省 load 指令/地址运算）——grid.x=nblk、无镜像配对、不做因果掩码。`--lsetma=0`
        //   退回 O68 的 cp.async 均衡版做同 binary A/B；`--lsefull=0` 仍退回 O1。
        bool did_tma = false;
#if defined(FA_WGMMA) && defined(FA_TMA)
        if (lse_tma) {
          launch_lse_bal_tma<128, 1, true>(lg, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                           scale);
          did_tma = true;
        }
#endif
        if (!did_tma)
          launch_lse_bal<128, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                       nullptr, nullptr, 1);
      } else
        launch_lse<128>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, (int)causal);
      if (do_delta) {
        if (delta_warp_sel)
          delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        else
          delta_kernel<128><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
      }
    } else if (D == 256) {
      // O76（第 171 轮）：head_dim=256 的 LSE。`lse_mma_kernel_bal` 对 HD 是模板参数
      //   （D=512 已验证），HD=256 直接复用其镜像配对（causal）/ 均衡 FULL（非 causal）几何。
      // O103（第 197 轮）：优先走 4D-TMA（对齐 D=128 O32 / D=512 O74）；`--lsetma=0` 退回
      //   O76/O101 的 mma/cp.async 版做同 binary A/B。causal 走镜像配对 `lg_bal` + split，
      //   full 走 `lg`（一个 m 块一个 CTA）+ split（FULL=true）。
      bool did_tma256 = false;
#if defined(FA_WGMMA) && defined(FA_TMA)
      if (lse_tma) {
        if (causal) {
          if (lse_split_eff > 1)
            launch_lse_bal_tma_split<256, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                             d_lse_part, S, H, Hkv, scale, lse_split_eff);
          else
            launch_lse_bal_tma<256, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                       scale);
        } else {
          if (lse_split_eff > 1)
            launch_lse_bal_tma_split<256, 1, true>(lg, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                                   d_lse_part, S, H, Hkv, scale, lse_split_eff);
          else
            launch_lse_bal_tma<256, 1, true>(lg, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                             scale);
        }
        did_tma256 = true;
      }
#endif
      if (!did_tma256) {
        if (causal)
          launch_lse_bal<256, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, nullptr,
                                 d_lse_part, lse_split_eff);
        else
          // O101（第 195 轮）：full D=256 的均衡 LSE 也接上 K 维 split（此前恒 split=1）。
          launch_lse_bal<256, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                       nullptr, d_lse_part, lse_split_eff);
      }
      if (do_delta) {
        if (delta_warp_sel)
          delta_warp_kernel<256><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        else
          delta_kernel<256><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
      }
    } else {
      // O39：D=512（MLA）causal LSE 也用 K 维 split（此前只有 D=128/TMA 有）。
      // O59：定长 causal MLA 默认走 O58 的 cfg6（PIPE1/LBN16，4 CTA/SM）；`--lseocc=4` 退回旧默认。
      if (causal) {
        // O74（第 169 轮）：MLA causal LSE 优先走 4D-TMA（`--lsetma=0` 退回 mma/cp.async A/B）。
        bool did_tma512 = false;
#if defined(FA_WGMMA) && defined(FA_TMA)
        if (lse_tma) {
          if (lse_split_eff > 1)
            launch_lse_bal_tma_split<512, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                             d_lse_part, S, H, Hkv, scale, lse_split_eff);
          else
            launch_lse_bal_tma<512, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                       scale);
          did_tma512 = true;
        }
#endif
        if (did_tma512) {
          // 已走 4D-TMA
        } else if (lseocc_opt == 5)
          launch_lse_bal<512, 0, false, 128, 32>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv,
                                                 scale, nullptr, d_lse_part, lse_split_eff);
        else if (lseocc_opt == 6 || causal_cfg6_fixed)
          launch_lse_bal<512, 1, false, 128, 16>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv,
                                                 scale, nullptr, d_lse_part, lse_split_eff);
        else
          launch_lse_bal<512, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, nullptr,
                                 d_lse_part, lse_split_eff);
      } else if (g_lse_full512_opt) {
        // O101（第 195 轮）：**定长 full MLA（D=512）的 LSE 补齐均衡 + cp.async 流水 + K 维 split**
        //   （对齐 O54 只修了的 varlen full 分支）。此前这里是 O1 的 `lse_mma_kernel`（无流水、
        //   无 split），实测占 full MLA 端到端 ~65%（S1024H2 preprocess 0.376ms > main 0.178ms）。
        //   `lse_mma_kernel_bal<512,PIPE,FULL=true>`：一个 CTA 一个 m 块（grid.x=nblk）、K 用
        //   `cp.async` 双缓冲；`lse_split_eff>1` 时把 grid.z 扩成 B*split 并跑 merge（只改 fp32
        //   求和次序）。`--lse512old=1` 退回 O1 做同 binary A/B。
        if (lse_split_eff > 1)
          launch_lse_bal<512, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                       nullptr, d_lse_part, lse_split_eff);
        else
          launch_lse_bal<512, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                       nullptr, nullptr, 1);
      } else
        launch_lse<512>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, (int)causal);
      if (do_delta) {
        if (delta_warp_sel)
          delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        else
          delta_kernel<512><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
      }
    }
  };

  // O12：LSE/D 预装寄存器（默认开），`--prel=0` 关；为同 session A/B 派发到两个模板实例。
  const bool prel_sel = (prel_opt < 0) ? true : (prel_opt != 0);
  // O7e-2：fold 16B 向量化写（默认开），`--f16b=0` 退回 O7e 的 4B 写（仅作 A/B）。
  const bool f16b_sel = (f16b_opt != 0);
  // O47：MLA（D=512）8-warp 几何（默认开；`--mla8w=0` 退回 4-warp A/B）。
  const bool mla8w_sel = (mla8w_opt < 0) ? true : (mla8w_opt != 0);
  // O48/O49：D=128 mma 路径的 8-warp 几何（opt-in / 自动）。默认 4-warp 与历史逐字相同。
  //   O49 自动档判据 = 「mma 后端（!wgmma）且有效 grid ≤ SM 数」；`--d128w=1` 仍可强制。
  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  const long fp8_grid = (long)mg.x * mg.y * mg.z;
  const bool d128w_sel =
      (d128w_opt > 0) ? true
                      : ((d128w_opt < 0) ? (D == 128 && !wgmma && fp8_grid <= sm_count)
                                         : false);
  if (D == 128) printf("[O49] d128 8-warp = %d (d128w=%d, grid=%ld, sm=%d, wgmma=%d)\n",
                       (int)d128w_sel, d128w_opt, fp8_grid, sm_count, wgmma);
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
    // O76（第 171 轮）：head_dim=256 走 mma 主 kernel（无 fp8 wgmma——它只做 HD=128；
    //   无 4D-TMA）。HD/NTW=2 ⇒ 不启用寄存器 dQ 累加（kRegDq 由 device 自动关）。
    //   与 D=512（MLA）同款 4-warp/128 线程几何；数值路径与其它 HD 逐字同源。
    // O84（第 179 轮）：放开 WGMMA——`-DFA_WGMMA` 构建下 D=256 默认走 wgmma 主 kernel
    //   （GEMM1/2 wgmma + SW128；K/V/dO 仍 cp.async 载入，非 TMA）。`--d256wgm=0` 退回 mma A/B。
    if (D == 256) {
#ifdef FA_WGMMA
      const bool d256_wg = (d256wgm_opt < 0) ? true : (d256wgm_opt != 0);
#if defined(FA_TMA)
      // O85（第 180 轮）：D=256 的 Q/dO 4D-TMA（chunk-major，NCH=2；K/V 仍 cp.async）。
      //   **A/B 判决为中性（±2%，见 docs/03 §108）⇒ 默认关**，`--d256tma=1` 复现。
      const bool d256_tma = (d256tma_opt > 0);
      if (d256_wg && d256_tma) {
        launch_bwd_main_qdtma<256, 64, 32, false>(
            mg, qmap_main, dmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
            d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        return;
      }
#endif
      if (d256_wg) {
        // O104：hswap256（跨 head 全局 LPT）——head 走快轴，grid=(H, nblk*ksplit, B)，配 O89 的
        //   `d_mrev`（LPT 反转表）⇒ 所有 head 的最贵 m 块一起先跑。只改调度，dK/dV 仍是可交换
        //   的跨 CTA 原子 ⇒ 数值只在 fp8 噪声内。`--hswap=0` 回退历史锯齿序。
        if (hswap256_elig) {
          dim3 mgh256(H, mg.x, mg.z);
          launch_bwd_main<256, 64, 32, false, true, true, true, true, THREADS, WN, true>(
              mgh256, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
          return;
        }
        // O105：大 base_grid（hswap256 跳过）时也透传 O89 的 per-head LPT 反转表 `d_mrev`
        //   （HSWAP=false：grid/head 排布不变，只有 m 块贵先跑）。`--mrev=0` 时 d_mrev=nullptr。
        launch_bwd_main<256, 64, 32, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                                  d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
                                                  d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit,
                                                  nullptr, nullptr, d_mrev);
        return;
      }
#endif
      launch_bwd_main<256, 64, 32, false>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
                                          d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                          scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
      return;
    }
    if (D == 128 && wg3) {
#ifdef FA_WGMMA
      // O91（F6-step4）：BM=192 / 3 warpgroup（384 线程）主 kernel，opt-in。
      launch_bwd_wgmma_nw<128, 192, 32, 3>(mg3, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                           d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc,
                                           S, H, Hkv, scale, (int)causal, ksplit3);
      return;
#else
      fprintf(stderr, "--wg3 需要 -DFA_WGMMA（sm_90a）构建\n");
      std::exit(2);
#endif
    }
    if (D == 128 && wg2wgmma) {
#ifdef FA_WGMMA
      launch_bwd_wgmma2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                             d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                             ksplit2);
      return;
#else
      fprintf(stderr, "--wg2wgmma 需要 -DFA_WGMMA（sm_90a）构建\n");
      std::exit(2);
#endif
    }
    if (D == 128 && wg2) {
      launch_bwd_wg2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                          d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                          ksplit2);
      return;
    }
#if defined(FA_WGMMA) && defined(FA_TMA)
    if (D == 128 && wg2tma) {
      // O90（F6-step3）：wgmma2（BM=128, 2 warpgroup）+ K/V 4D-TMA（K 双缓冲）。
      launch_bwd_wgmma2tma<128>(mg2, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                Hkv, scale, (int)causal, ksplit2);
      return;
    }
#endif
    if (D == 128 && d128w_sel) {
      // O48（候选 ①）：D=128 主 kernel 的 8-warp（256 线程 / 2×4 网格）几何。仅 mma 后端
      //   （`WGMMA=true` 的 GEMM1/2 是 warpgroup 级、static_assert 锁死 2 warp）。BN 可 32/64；
      //   `PREL/F16B/RCP` 取默认档（A/B 只关心 warp 几何）。默认 4-warp 路径不受影响。
      if (bn64_opt) {
        if (use_regdq)
          launch_bwd_main<128, 64, 64, true, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 64, false, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      } else {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      }
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
      // O93：跨 head 全局 LPT——grid 轴对调（head 走快轴）。仅当 O89 的 mrev 表已建
      //   （定长 causal D=128 nblk>=16）才有意义；否则维持历史锯齿序。
      if (hswap_opt && d_mrev) {
        // O95：`--ksm` 时 y 轴长度改为变-ks 槽位数，并把 slot_tab 传给主 kernel（消费 HSWAP 路径）。
        dim3 mgh(H, d_ksm ? (unsigned)ksm_nslots : mg.x, mg.z);
        if (use_regdq)
          launch_bwd_main_kvtma<128, 64, 32, true, true, true, true, true>(
              mgh, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
              d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
              scale, (int)causal, ksplit, nullptr, d_mrev, d_ksm);
        else
          launch_bwd_main_kvtma<128, 64, 32, false, true, true, true, true>(
              mgh, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
              d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
              scale, (int)causal, ksplit, nullptr, d_mrev, d_ksm);
        return;
      }
      if (use_regdq)
        launch_bwd_main_kvtma<128, 64, 32, true>(
            mg, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
            d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
            scale, (int)causal, ksplit, nullptr, d_mrev);
      else
        launch_bwd_main_kvtma<128, 64, 32, false>(
            mg, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
            d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
            scale, (int)causal, ksplit, nullptr, d_mrev);
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
    // O47：MLA（D=512）默认走 8-warp/256 线程几何（`--mla8w=0` 退回 4-warp 供 A/B）。
    // O51：8-warp 档下 K/V 默认走 cp.async 回填流水（`--mlakvp=0` 退回同步载入 A/B）。
    // O105：定长 causal MLA 也透传 O89 的 per-head LPT 反转表 `d_mrev`（grid/head 排布不变）。
    const bool mla_kvp_sel = (mla_kvp_opt < 0) ? true : (mla_kvp_opt != 0);
    if (mla8w_sel) {
      if (prel_sel && mla_kvp_sel)
        launch_bwd_main_kvpipe<512, 64, 32, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
      else if (prel_sel)
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
      else
        launch_bwd_main<512, 64, 32, false, false, false, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
    } else if (prel_sel)
      launch_bwd_main<512, 64, 32, false, false, true, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                       d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                       (int)causal, ksplit, nullptr, nullptr, d_mrev);
    else
      launch_bwd_main<512, 64, 32, false, false, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                        d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                        (int)causal, ksplit, nullptr, nullptr, d_mrev);
  };

  // ---- O78：端到端 overlap（按 head 分块，LSE 与 main 跨 stream 流水）的设置。----
  const bool ovlp_req = (ovlp >= 2) && (D == 128) && causal && (Hkv == H) && (H % ovlp == 0) &&
                        qfuse && qfast && dfuse && delta_warp_sel;
  int ovlp_n = ovlp_req ? ovlp : 0;
  const int ovlp_hc = ovlp_n ? (H / ovlp_n) : H;
  bool ovlp_path = false;
  std::vector<CUtensorMap> c_q, c_d, c_k, c_v, c_lq, c_lk;
  cudaStream_t ovA = nullptr, ovB = nullptr;
  std::vector<cudaEvent_t> ov_eL(ovlp_n), ov_eM(ovlp_n);
  cudaEvent_t ov_eQ = nullptr;
  CUDA_CHECK(cudaEventCreateWithFlags(&ov_eQ, cudaEventDisableTiming));
#if defined(FA_WGMMA) && defined(FA_TMA)
  {
    const bool rcp_sel0 = (foldrcp_opt != 0);
    if (ovlp_n && wgmma && qd_tma && kv_tma && prel_sel && f16b_sel && rcp_sel0) {
      ovlp_path = true;
      for (int k = 0; k < ovlp_n; ++k) {
        const long long h0 = (long long)k * ovlp_hc;
        c_q.push_back(make_map_fp8_chunk(d_q8, H, h0, ovlp_hc, S, D, B, 64));
        c_d.push_back(make_map_fp8_chunk(d_do8, H, h0, ovlp_hc, S, D, B, 64));
        c_k.push_back(make_map_fp8_chunk(d_k8, Hkv, h0, ovlp_hc, S, D, B, 32));
        c_v.push_back(make_map_fp8_chunk(d_v8, Hkv, h0, ovlp_hc, S, D, B, 32));
        c_lq.push_back(make_map_fp8_chunk(d_q8, H, h0, ovlp_hc, S, D, B, 64));
        c_lk.push_back(make_map_fp8_chunk(d_k8, Hkv, h0, ovlp_hc, S, D, B, 64));
      }
      CUDA_CHECK(cudaStreamCreateWithFlags(&ovA, cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamCreateWithFlags(&ovB, cudaStreamNonBlocking));
      for (int k = 0; k < ovlp_n; ++k) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ov_eL[k], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ov_eM[k], cudaEventDisableTiming));
      }
      printf("O78: overlap ON, head chunks=%d (hc=%d), LSE(sA) || main(sB)\n", ovlp_n, ovlp_hc);
    } else if (ovlp >= 2) {
      printf("O78: overlap requested but conditions unmet (D=%d causal=%d Hkv=%d H=%d "
             "wgmma=%d qd_tma=%d kv_tma=%d) -> serial\n",
             D, (int)causal, Hkv, H, wgmma, qd_tma, kv_tma);
    }
  }
#endif

  auto run_all = [&]() {
    if (qfuse && qfast) {
      // O66：默认把 delta 也融进 quant kernel（`dfuse && delta_warp_sel`）。
      if (dfuse && delta_warp_sel)
        quant_zero_delta();
      else
        quant_zero();   // O64：4 次量化 + 3 次清零融合为 1 个 launch
    } else {
      quant();
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    }
    if (ovlp_path) {
#if defined(FA_WGMMA) && defined(FA_TMA)
      // O78：按 head 分块，LSE(sA) 与 main(sB) 流水。quant（含 delta/清零）已在 default stream
      //   整体完成，两条 stream 先等 `ov_eQ`。
      CUDA_CHECK(cudaEventRecord(ov_eQ));
      CUDA_CHECK(cudaStreamWaitEvent(ovA, ov_eQ, 0));
      CUDA_CHECK(cudaStreamWaitEvent(ovB, ov_eQ, 0));
      for (int k = 0; k < ovlp_n; ++k) {
        const long long h0 = (long long)k * ovlp_hc;
        dim3 lgc(lg_bal.x, (unsigned)ovlp_hc, B);
        if (!ovl_nolse) {
          launch_lse_bal_tma<128, 1>(lgc, c_lq[k], c_lk[k], d_qs + h0, d_ks + h0, d_lse + h0,
                                     S, H, Hkv, scale, nullptr, ovA);
          CUDA_CHECK(cudaEventRecord(ov_eL[k], ovA));
          CUDA_CHECK(cudaStreamWaitEvent(ovB, ov_eL[k], 0));
        }
        dim3 mgc(mg.x, (unsigned)ovlp_hc, mg.z);
        if (use_regdq)
          launch_bwd_main_kvtma<128, 64, 32, true>(
              mgc, c_q[k], c_d[k], c_k[k], c_v[k], d_q8 + (size_t)h0 * D, d_qs + h0,
              d_k8 + (size_t)h0 * D, d_ks + h0, d_v8 + (size_t)h0 * D, d_vs + h0,
              d_do8 + (size_t)h0 * D, d_dos + h0, d_delta + h0, d_lse + h0,
              d_dq_acc + (size_t)h0 * D, d_dk_acc + (size_t)h0 * D, d_dv_acc + (size_t)h0 * D,
              S, H, Hkv, scale, (int)causal, ksplit, ovB);
        else
          launch_bwd_main_kvtma<128, 64, 32, false>(
              mgc, c_q[k], c_d[k], c_k[k], c_v[k], d_q8 + (size_t)h0 * D, d_qs + h0,
              d_k8 + (size_t)h0 * D, d_ks + h0, d_v8 + (size_t)h0 * D, d_vs + h0,
              d_do8 + (size_t)h0 * D, d_dos + h0, d_delta + h0, d_lse + h0,
              d_dq_acc + (size_t)h0 * D, d_dk_acc + (size_t)h0 * D, d_dv_acc + (size_t)h0 * D,
              S, H, Hkv, scale, (int)causal, ksplit, ovB);
        CUDA_CHECK(cudaEventRecord(ov_eM[k], ovB));
      }
      for (int k = 0; k < ovlp_n; ++k) CUDA_CHECK(cudaStreamWaitEvent(nullptr, ov_eM[k], 0));
      if (cvt_on)
        convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                    d_dv, nq, nkv);
#endif
      return;
    }
    // O66：delta 已在 quant 阶段算好时，preprocess 跳过独立 delta launch。
    run_preprocess(!(qfuse && qfast && dfuse && delta_warp_sel));
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
  if (qfuse && qfast) {
    if (dfuse && delta_warp_sel)
      for (int i = 0; i < iters; ++i) quant_zero_delta();
    else
      for (int i = 0; i < iters; ++i) quant_zero();
  } else
    for (int i = 0; i < iters; ++i) quant();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_quant = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_quant, ev0, ev1));
  ms_quant /= iters;

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i)
    run_preprocess(!(qfuse && qfast && dfuse && delta_warp_sel));
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
  printf("[timing] quant %.4f ms | preprocess %.4f ms | main %.4f ms | convert %.4f ms (cvt_on=%d, qfuse=%d, dfuse=%d)\n",
         ms_quant, ms_pre, ms_main, ms - ms_quant - ms_pre - ms_main, cvt_on, qfuse, dfuse);

  // ---- O78（实验/诊断）：main 与 LSE 并发（两条非阻塞 stream）vs 串行的墙钟对比。
  //   目的：判定「端到端 overlap（preprocess/main 跨 head 流水）」在本卡是否可行。
  //   只计时、不改结果（并发期的 LSE 写与 main 读 d_lse 竞争，输出丢弃）。----
  if (ovltest && D == 128 && causal) {
#if defined(FA_WGMMA) && defined(FA_TMA)
    const bool rcp_sel = (foldrcp_opt != 0);
    if (wgmma && qd_tma && kv_tma && prel_sel && f16b_sel && rcp_sel) {
      cudaStream_t sA, sB;
      CUDA_CHECK(cudaStreamCreateWithFlags(&sA, cudaStreamNonBlocking));
      CUDA_CHECK(cudaStreamCreateWithFlags(&sB, cudaStreamNonBlocking));
      auto lse_s = [&](cudaStream_t s) {
        if (lse_split_eff > 1)
          launch_lse_bal_tma_split<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                           d_lse_part, S, H, Hkv, scale, lse_split_eff, s);
        else
          launch_lse_bal_tma<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                     scale, nullptr, s);
      };
      auto main_s = [&](cudaStream_t s) {
        if (use_regdq)
          launch_bwd_main_kvtma<128, 64, 32, true>(mg, qmap_main, dmap_main, kmap_main, vmap_main,
              d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, s);
        else
          launch_bwd_main_kvtma<128, 64, 32, false>(mg, qmap_main, dmap_main, kmap_main, vmap_main,
              d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, s);
      };
      for (int i = 0; i < 3; ++i) { main_s(sA); lse_s(sB); }
      CUDA_CHECK(cudaStreamSynchronize(sA));
      CUDA_CHECK(cudaStreamSynchronize(sB));
      cudaEvent_t e0, e1, eA, eB;
      CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
      CUDA_CHECK(cudaEventCreate(&eA)); CUDA_CHECK(cudaEventCreate(&eB));
      float t_main = 0.f, t_lse = 0.f, wall = 0.f;
      CUDA_CHECK(cudaEventRecord(e0, sA));
      for (int i = 0; i < iters; ++i) main_s(sA);
      CUDA_CHECK(cudaEventRecord(e1, sA));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_main, e0, e1)); t_main /= iters;
      CUDA_CHECK(cudaEventRecord(e0, sB));
      for (int i = 0; i < iters; ++i) lse_s(sB);
      CUDA_CHECK(cudaEventRecord(e1, sB));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&t_lse, e0, e1)); t_lse /= iters;
      // 并发：两 stream 同时发（eA=main 完、eB=lse 完），sA 等 eB 后记 e1 ⇒ wall 覆盖两者。
      CUDA_CHECK(cudaEventRecord(e0, sA));
      for (int i = 0; i < iters; ++i) main_s(sA);
      CUDA_CHECK(cudaEventRecord(eA, sA));
      for (int i = 0; i < iters; ++i) lse_s(sB);
      CUDA_CHECK(cudaEventRecord(eB, sB));
      CUDA_CHECK(cudaStreamWaitEvent(sA, eB, 0));
      CUDA_CHECK(cudaEventRecord(e1, sA));
      CUDA_CHECK(cudaEventSynchronize(e1));
      CUDA_CHECK(cudaEventElapsedTime(&wall, e0, e1)); wall /= iters;
      float tm_a = 0.f, tb = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&tm_a, e0, eA)); tm_a /= iters;
      CUDA_CHECK(cudaEventElapsedTime(&tb, eA, e1)); tb /= iters;
      printf("[O78 ovltest] main=%.4f ms | lse=%.4f ms | serial(sum)=%.4f ms | concurrent wall=%.4f ms"
             " (main_end@%.4f) => overlap %.3fx (concurrent/serial)\n",
             t_main, t_lse, t_main + t_lse, wall, tm_a, wall / (t_main + t_lse));
      cudaStreamDestroy(sA); cudaStreamDestroy(sB);
    }
#endif
  }

  // ---- O64 A/B：融合 quant+zero 的同一 binary 端到端对比。----
  {
    auto time_all = [&](bool f) {
      qfuse = f ? 1 : 0;
      for (int i = 0; i < 3; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      return t / iters;
    };
    const float t_f = time_all(true), t_u = time_all(false);
    // 数值检查：两模式的输出差异应只来自跨 CTA `atomicAdd` 的次序（~e-7）。
    std::vector<float> a(nq + 2 * nkv), b(nq + 2 * nkv);
    qfuse = 1; run_all();
    CUDA_CHECK(cudaMemcpy(a.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a.data() + nq, d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a.data() + nq + nkv, d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    qfuse = 0; run_all();
    CUDA_CHECK(cudaMemcpy(b.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data() + nq, d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data() + nq + nkv, d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    qfuse = 1;
    double md = 0, mq = 0, mk = 0, mv = 0;
    for (size_t i = 0; i < nq; ++i) mq = std::max(mq, (double)fabsf(a[i] - b[i]));
    for (size_t i = 0; i < nkv; ++i) mk = std::max(mk, (double)fabsf(a[nq + i] - b[nq + i]));
    for (size_t i = 0; i < nkv; ++i) mv = std::max(mv, (double)fabsf(a[nq + nkv + i] - b[nq + nkv + i]));
    md = std::max(mq, std::max(mk, mv));
    printf("[O64 A/B] end2end fused(1 launch) %.4f ms | unfused(4 quant + 3 memset) %.4f ms "
           "(%.4fx) | max_abs(fused-vs-unfused) dq/dk/dv=%.3e/%.3e/%.3e\n",
           t_f, t_u, t_u / t_f, mq, mk, mv);
  }

  // ---- O66 A/B：把 delta 融进 quant kernel（同一 binary）端到端对比 + delta 逐位校验。----
  {
    auto time_all = [&](bool f) {
      dfuse = f ? 1 : 0;
      for (int i = 0; i < 3; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      return t / iters;
    };
    const float t_f = time_all(true), t_u = time_all(false);
    std::vector<float> df((size_t)d_rows), du((size_t)d_rows);
    dfuse = 1; run_all();
    CUDA_CHECK(cudaMemcpy(df.data(), d_delta, (size_t)d_rows * 4, cudaMemcpyDeviceToHost));
    dfuse = 0; run_all();
    CUDA_CHECK(cudaMemcpy(du.data(), d_delta, (size_t)d_rows * 4, cudaMemcpyDeviceToHost));
    dfuse = 1;
    double mdelta = 0;
    for (int i = 0; i < d_rows; ++i) mdelta = std::max(mdelta, (double)fabsf(df[i] - du[i]));
    printf("[O66 A/B] end2end delta-fused %.4f ms | delta-separate %.4f ms (%.4fx) | "
           "max_abs(delta fused-vs-separate)=%.3e\n", t_f, t_u, t_u / t_f, mdelta);
  }

  // ---- O48 A/B（D=128，仅 mma 路径）：主 kernel 4-warp（128/2）vs 8-warp（256/4 网格）。
  //      同 session 计时 + 逐元素对拍（只换 warp 网格、数学/数据流不变）。
  if (D == 128 && !bn64_opt) {
    auto run_d128w = [&](int nth) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (nth == 256) {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false, true, true, true, 256, 4>(
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
    auto bench_d128w = [&](int nth, float* out) {
      run_d128w(nth);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_d128w(nth);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float m4w = 0.f, m8w = 0.f;
    bench_d128w(128, &m4w);
    bench_d128w(256, &m8w);
    std::vector<float> a4_dq(nq), a4_dk(nkv), a4_dv(nkv), b8_dq(nq), b8_dk(nkv), b8_dv(nkv);
    run_d128w(128);
    CUDA_CHECK(cudaMemcpy(a4_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a4_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a4_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_d128w(256);
    CUDA_CHECK(cudaMemcpy(b8_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b8_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b8_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto md = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O48 A/B] main D=128 4w(128/2) %.4f ms | 8w(256/4) %.4f ms (%.3fx) | "
           "max_abs(8w-vs-4w) dq/dk/dv=%.3e/%.3e/%.3e\n",
           m4w, m8w, m4w / m8w, md(b8_dq, a4_dq), md(b8_dk, a4_dk), md(b8_dv, a4_dv));
    run_main();   // 恢复最终输出为 CLI 选中的路径
  }

  // ---- P3-4e/P3-4f A/B（D=128 定长，`--det` / `--detk=N`）：跨 CTA `atomicAdd`（非确定性）
  //      vs 确定性 partial + 固定次序二次归约（对齐 fp16/bf16 O7b）。`--detk>1` 时 dQ 也走
  //      partial（`dq_reduce_kernel`），验证「确定性 + split-K 并行度」。同 session 计时 +
  //      跑两遍 DET 验证逐位可复现，再与 atomic 比 max|diff|。BK=64（fp8 默认 BM）。
  //      注：dK/dV 的 partial 天然无需 part 维——每个 (mblk, jg) 的 K tile 只属于一个 part，
  //      各 part 写不相交的 jg；跨 part 的贡献经固定次序 mblk 归约。----
  if (det_ab && D == 128 && !varlen) {
    const int nblk_d = (S + 63) / 64;
    // O102（第 196 轮）：DET 的 auto ksplit 标定——causal 下 k=4 触底（P3-4f 已测；
    //   full 无三角偏斜、多切只增 Q/dO 重读 + dQ 跨 part 原子 ⇒ k=1）。`--detk=K` 仍可覆盖。
    const int ks = det_ksplit >= 1 ? det_ksplit : (causal ? (nblk_d < 4 ? nblk_d : 4) : 1);
    const size_t part_elems = (size_t)B * H * nblk_d * S * D;
    const size_t dqpart_elems = (size_t)B * S * H * (size_t)ks * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    float* d_dq_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
    auto run_at = [&](int k) {
      dim3 g((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_main<128, 64, 32, true, false, true, true, true>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
          d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, k);
    };
    auto run_dt = [&](int k) {
      dim3 g((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
      launch_bwd_main_det<128, 64, 32, true>(g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                            d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc,
                                            S, H, Hkv, scale, (int)causal, k, d_dk_part, d_dv_part,
                                            nblk_d, d_dq_part);
      // P3-4n：dkv+dq 融合归约（默认）vs 两次 launch（`--nofusered`）。
      const int dkv_blocks = B * Hkv * S;
      const int dq_blocks = (k > 1) ? B * S * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_kernel<128, 64><<<dkv_blocks + dq_blocks, 128>>>(
            d_dk_part, d_dv_part, (k > 1) ? d_dq_part : nullptr, d_dk_acc, d_dv_acc, d_dq_acc, S,
            H, Hkv, nblk_d, (int)causal, k, dkv_blocks);
      } else {
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H,
                                                Hkv, nblk_d, (int)causal);
        if (k > 1) {
          dim3 dg(B * S, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq_acc, S, H, k);
        }
      }
    };
    auto time_fn2 = [&](auto fn, float* out_ms) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_at = 0.f, ms_dt = 0.f;
    time_fn2([&] { run_at(ks); }, &ms_at);
    std::vector<float> ak(nkv), av(nkv), aq(nq);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    time_fn2([&] { run_dt(ks); }, &ms_dt);
    std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    run_dt(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    auto mad2 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[P3-4f A/B] ksplit=%d | atomic %.4f ms | DET(partial+reduce) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
           ks, ms_at, ms_dt, ms_at / ms_dt, mad2(dq1, dq2), mad2(dk1, dk2), mad2(dv1, dv2),
           mad2(dq1, aq), mad2(dk1, ak), mad2(dv1, av));
    // ---- P3-4n A/B：二次归约「两次 launch（分离）」vs「一次 launch（融合）」，只测 reduce。
    //      partial 先用一次 DET 主 kernel 预置好；两版只差网格组织、求和次序逐字相同。----
    {
      auto only_sep = [&]() {
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H,
                                                Hkv, nblk_d, (int)causal);
        if (ks > 1) {
          dim3 dg(B * S, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq_acc, S, H, ks);
        }
      };
      auto only_fus = [&]() {
        const int dkv_blocks = B * Hkv * S;
        const int dq_blocks = (ks > 1) ? B * S * H : 0;
        dkv_dq_reduce_kernel<128, 64><<<dkv_blocks + dq_blocks, 128>>>(
            d_dk_part, d_dv_part, (ks > 1) ? d_dq_part : nullptr, d_dk_acc, d_dv_acc, d_dq_acc, S,
            H, Hkv, nblk_d, (int)causal, ks, dkv_blocks);
      };
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      {
        dim3 g((S + 63) / 64 * ks, H, B);
        launch_bwd_main_det<128, 64, 32, true>(g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                               d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc,
                                               S, H, Hkv, scale, (int)causal, ks, d_dk_part,
                                               d_dv_part, nblk_d, d_dq_part);
      }
      float t_rs = 0.f, t_rf = 0.f;
      time_fn2(only_sep, &t_rs);
      std::vector<float> rsdk(nkv), rsdv(nkv), rsdq(nq);
      CUDA_CHECK(cudaMemcpy(rsdk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rsdv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rsdq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      time_fn2(only_fus, &t_rf);
      std::vector<float> rfdk(nkv), rfdv(nkv), rfdq(nq);
      CUDA_CHECK(cudaMemcpy(rfdk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rfdv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rfdq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4n A/B] reduce separate(2 launch) %.4f ms | fused(1 launch) %.4f ms (%.3fx) | "
             "fused-vs-sep bitwise dq/dk/dv=%.2e/%.2e/%.2e\n",
             t_rs, t_rf, t_rs / t_rf, mad2(rfdq, rsdq), mad2(rfdk, rsdk), mad2(rfdv, rsdv));
    }
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
    if (d_dq_part) cudaFree(d_dq_part);
    run_main();   // 恢复最终输出为 CLI 选中的路径
  }

  // ---- P3-4h/P3-4k A/B（D=512 MLA 定长，`--det` / `--detk=N`）：把确定性 dK/dV 从 D=128
  //      扩到 MLA（HD=512）。P3-4e/f 的 partial 布局与 `dkv_reduce_kernel` 本就对任意 HD 成立，
  //      body 的 DET 分支也是 HD 无关的；故只需给 `launch_bwd_main_det` 加 NTH/NWAR 模板参数、
  //      在 host 补一条 MLA 的 A/B。atomic 参照用同 256/4 几何的非 kvpipe 主 kernel（仅 DET
  //      一个变量不同）。跑两遍 DET 验证逐位可复现。----
  //      P3-4k：MLA 的 `kRegDq` 恒 false（HD/NTW=4）⇒ 旧版锁 ksplit=1（单写者 `red_add2` 保
  //      dQ 确定）。本轮把非 `kRegDq` 的 dQ epilogue 在 `DET && ksplit>1` 时改写到 `dq_part`
  //      的 part 分片 + `dq_reduce_kernel` 固定次序求和，从而**在 ksplit>1 下也确定**、恢复
  //      split-K 并行度。`--detk=N` 触发；默认 `--det`（ks=1）与 P3-4h 逐位一致。----
  if (det_ab && D == 512 && !varlen) {
    const int nblk_d = (S + 63) / 64;
    const int ks = det_ksplit < 1 ? 1 : det_ksplit;
    const size_t part_elems = (size_t)B * H * nblk_d * S * D;
    const size_t dqpart_elems = (size_t)B * S * H * (size_t)ks * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    float* d_dq_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
    auto run_at = [&](int k) {
      dim3 g1((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
          g1, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
          d_dv_acc, S, H, Hkv, scale, (int)causal, k);
    };
    auto do_reduce = [&](int k) {
      const int dkv_blocks = B * Hkv * S;
      const int dq_blocks = (k > 1) ? B * S * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_kernel<512, 64><<<dkv_blocks + dq_blocks, 512>>>(
            d_dk_part, d_dv_part, (k > 1) ? d_dq_part : nullptr, d_dk_acc, d_dv_acc, d_dq_acc, S,
            H, Hkv, nblk_d, (int)causal, k, dkv_blocks);
      } else {
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<512, 64><<<rg, 512>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H,
                                                Hkv, nblk_d, (int)causal);
        if (k > 1) {
          dim3 dg(B * S, H);
          dq_reduce_kernel<512><<<dg, 512>>>(d_dq_part, d_dq_acc, S, H, k);
        }
      }
    };
    auto run_dt = [&](int k) {
      dim3 g1((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4>(
          g1, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
          d_dv_acc, S, H, Hkv, scale, (int)causal, k, d_dk_part, d_dv_part, nblk_d, d_dq_part);
      do_reduce(k);
    };
    // P3-4m：DET + O51 K/V `cp.async` 回填流水（kvpipe），数值应与非 kvpipe 逐位相同。
    auto run_dt_kv = [&](int k) {
      dim3 g1((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4, false, true>(
          g1, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
          d_dv_acc, S, H, Hkv, scale, (int)causal, k, d_dk_part, d_dv_part, nblk_d, d_dq_part);
      do_reduce(k);
    };
    auto tm2h = [&](auto fn, float* o) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(o, ev0, ev1));
      *o /= iters;
    };
    auto mad2h = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    float t_at = 0.f, t_dt = 0.f, t_dt1 = 0.f, t_dt_kv = 0.f;
    tm2h([&] { run_at(ks); }, &t_at);
    std::vector<float> ak(nkv), av(nkv), aq(nq);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    tm2h([&] { run_dt(1); }, &t_dt1);
    tm2h([&] { run_dt(ks); }, &t_dt);
    std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    run_dt(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    // P3-4m：DET + kvpipe（数值应与非 kvpipe 逐位相同）。
    tm2h([&] { run_dt_kv(ks); }, &t_dt_kv);
    std::vector<float> dkk1(nkv), dvk1(nkv), dqk1(nq), dkk2(nkv), dvk2(nkv), dqk2(nq);
    CUDA_CHECK(cudaMemcpy(dkk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvk1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dqk1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    run_dt_kv(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dkk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvk2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dqk2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    printf("[P3-4k A/B] MLA ksplit=%d atomic(256/4) %.4f ms | DET %.4f ms (vs atomic %.3fx) | "
           "DET k=1 %.4f ms -> k=%d %.4f ms (%.3fx split speedup) | "
           "runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
           ks, t_at, t_dt, t_at / t_dt, t_dt1, ks, t_dt, t_dt1 / t_dt, mad2h(dq1, dq2),
           mad2h(dk1, dk2), mad2h(dv1, dv2), mad2h(dq1, aq), mad2h(dk1, ak), mad2h(dv1, av));
    printf("[P3-4m A/B] MLA ksplit=%d | DET nonkv %.4f ms | DET kvpipe %.4f ms (%.3fx) | "
           "kvpipe runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | kvpipe-vs-nonkv dq/dk/dv="
           "%.2e/%.2e/%.2e\n",
           ks, t_dt, t_dt_kv, t_dt / t_dt_kv, mad2h(dqk1, dqk2), mad2h(dkk1, dkk2),
           mad2h(dvk1, dvk2), mad2h(dqk1, dq1), mad2h(dkk1, dk1), mad2h(dvk1, dv1));
    // ---- F4-b（第 134 轮）：MLA（HD=512）定长的「fp16 partial + O62 扇区化」。
    //      `dkv_reduce_kernel<512,64,true>` 已支持 P16；此处只把主 kernel 换 `DET_HALF`。----
    __half* d_dk_part_h = nullptr;
    __half* d_dv_part_h = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part_h, part_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dv_part_h, part_elems * sizeof(__half)));
    auto run_dth = [&](int k) {
      dim3 g1((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4, false, false, true>(
          g1, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
          d_dv_acc, S, H, Hkv, scale, (int)causal, k, reinterpret_cast<float*>(d_dk_part_h),
          reinterpret_cast<float*>(d_dv_part_h), nblk_d, d_dq_part);
      const int dkv_blocks = B * Hkv * ((S + 3) / 4);  // F4-c：P16 归约每 block 4 行
      const int dq_blocks = (k > 1) ? B * S * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_kernel<512, 64, true><<<dkv_blocks + dq_blocks, 512>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), (k > 1) ? d_dq_part : nullptr, d_dk_acc,
            d_dv_acc, d_dq_acc, S, H, Hkv, nblk_d, (int)causal, k, dkv_blocks);
      } else {
        dim3 rg(B * Hkv, (S + 3) / 4);  // F4-c：P16 归约每 block 4 行
        dkv_reduce_kernel<512, 64, true><<<rg, 512>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), d_dk_acc, d_dv_acc, S, H, Hkv, nblk_d,
            (int)causal);
        if (k > 1) {
          dim3 dg(B * S, H);
          dq_reduce_kernel<512><<<dg, 512>>>(d_dq_part, d_dq_acc, S, H, k);
        }
      }
    };
    float t_dth = 0.f;
    tm2h([&] { run_dth(ks); }, &t_dth);
    std::vector<float> hdk1(nkv), hdv1(nkv), hdk2(nkv), hdv2(nkv);
    CUDA_CHECK(cudaMemcpy(hdk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_dth(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[F4 A/B] MLA ksplit=%d | DET-fp32 %.4f ms | DET-fp16(扇区化) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dk/dv=%.2e/%.2e | fp16-vs-fp32 dk/dv=%.2e/%.2e\n",
           ks, t_dt, t_dth, t_dt / t_dth, mad2h(hdk1, hdk2), mad2h(hdv1, hdv2),
           mad2h(hdk1, dk1), mad2h(hdv1, dv1));
    cudaFree(d_dk_part_h);
    cudaFree(d_dv_part_h);
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
    if (d_dq_part) cudaFree(d_dq_part);
    run_main();   // 恢复最终输出为 CLI 选中的路径
  }

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
    // ---- P3-4g/O102 A/B（D=128/Hopper TMA，`--det`）：把确定性 dK/dV 从默认 mma 路径扩到
    //      Q/dO/K/V-TMA 快路。P3-4g 原版锁 ksplit=1；O102（第 196 轮）放开 `--detk=N>1`——
    //      dQ 走 per-part `dq_part` + `dq_reduce_kernel`，恢复 split-K 并行度（同 mma 路径
    //      P3-4f）。跑两遍 DET 验证 dq/dk/dv 逐位可复现，再与 atomic 比 max|diff|。----
    if (det_ab && !varlen) {
      const int nblk_d = (S + 63) / 64;
      // O102：DET auto ksplit（causal k=4 触底 / full k=1）——Hopper 快路同 mma 路径标定。
      const int ks = det_ksplit >= 1 ? det_ksplit : (causal ? (nblk_d < 4 ? nblk_d : 4) : 1);
      const size_t part_elems = (size_t)B * H * nblk_d * S * D;
      const size_t dqpart_elems = (size_t)B * S * H * (size_t)ks * D;
      float* d_dk_part = nullptr;
      float* d_dv_part = nullptr;
      float* d_dk_part_h = nullptr;
      float* d_dv_part_h = nullptr;
      float* d_dq_part = nullptr;
      CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
      // F4：fp16 partial（字节减半 + O62 扇区化）。元素数不变，仅存储类型改为 half。
      CUDA_CHECK(cudaMalloc(&d_dk_part_h, part_elems * sizeof(__half)));
      CUDA_CHECK(cudaMalloc(&d_dv_part_h, part_elems * sizeof(__half)));
      if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
      dim3 g1((S + 63) / 64 * ks, H, B);
      auto dq_reduce = [&]() {
        if (ks > 1)
          dq_reduce_kernel<128><<<dim3(B * S, H), 128>>>(d_dq_part, d_dq_acc, S, H, ks);
      };
      auto run_at = [&]() {
        CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
        CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
        CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
        if (use_regdq)
          launch_bwd_main_kvtma<128, 64, 32, true>(g1, qmap_main, dmap_main, kmap_main,
                                                   vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                   d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                   (int)causal, ks);
        else
          launch_bwd_main_kvtma<128, 64, 32, false>(g1, qmap_main, dmap_main, kmap_main,
                                                    vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                    d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                    d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                    (int)causal, ks);
      };
      auto run_dt = [&]() {
        CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
        CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
        CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
        if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
        if (use_regdq)
          launch_bwd_main_kvtma_det<128, 64, 32, true>(g1, qmap_main, dmap_main, kmap_main,
                                                       vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                       d_vs, d_do8, d_dos, d_delta, d_lse,
                                                       d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                       scale, (int)causal, d_dk_part, d_dv_part,
                                                       nblk_d, ks, d_dq_part);
        else
          launch_bwd_main_kvtma_det<128, 64, 32, false>(g1, qmap_main, dmap_main, kmap_main,
                                                        vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                        d_vs, d_do8, d_dos, d_delta, d_lse,
                                                        d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                        scale, (int)causal, d_dk_part, d_dv_part,
                                                        nblk_d, ks, d_dq_part);
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                nblk_d, (int)causal);
        dq_reduce();
      };
      // F4：fp16 partial + O62 扇区化的 DET（写侧与归约侧都走 P16 布局）。
      auto run_dth = [&]() {
        CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
        CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
        CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
        if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
        if (use_regdq)
          launch_bwd_main_kvtma_det<128, 64, 32, true, true, true, true, true>(
              g1, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
              d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
              (int)causal, d_dk_part_h, d_dv_part_h, nblk_d, ks, d_dq_part);
        else
          launch_bwd_main_kvtma_det<128, 64, 32, false, true, true, true, true>(
              g1, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
              d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
              (int)causal, d_dk_part_h, d_dv_part_h, nblk_d, ks, d_dq_part);
        dim3 rg(B * Hkv, (S + 3) / 4);  // F4-c：P16 归约每 block 4 行
        dkv_reduce_kernel<128, 64, true><<<rg, 128>>>(d_dk_part_h, d_dv_part_h, d_dk_acc, d_dv_acc,
                                                      S, H, Hkv, nblk_d, (int)causal);
        dq_reduce();
      };
      auto tm2 = [&](auto fn, float* o) {
        for (int i = 0; i < 3; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(o, ev0, ev1));
        *o /= iters;
      };
      float t_at = 0.f, t_dt = 0.f, t_dth = 0.f;
      tm2(run_at, &t_at);
      std::vector<float> ak(nkv), av(nkv), aq(nq);
      CUDA_CHECK(cudaMemcpy(ak.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(av.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(aq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      tm2(run_dt, &t_dt);
      std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
      CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      run_dt();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      tm2(run_dth, &t_dth);
      std::vector<float> hk1(nkv), hv1(nkv), hq1(nq), hk2(nkv), hv2(nkv), hq2(nq);
      CUDA_CHECK(cudaMemcpy(hk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hq1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      run_dth();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hq2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4g A/B] kvtma ksplit=%d atomic %.4f ms | DET %.4f ms (%.3fx) | "
             "runs[1-2] dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
             ks, t_at, t_dt, t_at / t_dt, maxd(dq1, dq2), maxd(dk1, dk2), maxd(dv1, dv2),
             maxd(dq1, aq), maxd(dk1, ak), maxd(dv1, av));
      printf("[F4 A/B] kvtma ksplit=%d atomic %.4f ms | DET-fp32 %.4f ms (%.3fx) | "
             "DET-fp16(扇区化) %.4f ms (%.3fx vs atomic) | runs[1-2] dq/dk/dv=%.2e/%.2e/%.2e | "
             "fp16-vs-fp32 dq/dk/dv=%.2e/%.2e/%.2e | fp16-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
             ks, t_at, t_dt, t_at / t_dt, t_dth, t_at / t_dth, maxd(hq1, hq2), maxd(hk1, hk2),
             maxd(hv1, hv2), maxd(hq1, dq1), maxd(hk1, dk1), maxd(hv1, dv1), maxd(hq1, aq),
             maxd(hk1, ak), maxd(hv1, av));
      cudaFree(d_dk_part);
      cudaFree(d_dv_part);
      cudaFree(d_dk_part_h);
      cudaFree(d_dv_part_h);
      if (d_dq_part) cudaFree(d_dq_part);
    }
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

  // ---- O59 A/B（D=512/MLA/causal/定长）：旧默认 LSE(`<512,1>` PIPE1/LBN64) vs O58 的 cfg6
  //   (`<512,1,false,128,16>` PIPE1/LBN16，4 CTA/SM)。只改 LSE 的 smem 几何/并行度，数学口径
  //   不变（split 只改 fp32 求和次序）。同 session、LSE-only 计时 + 逐元素对拍。----
  if (causal && D == 512) {
    auto auto_split = [&](int target) {
      long lg_grid = (long)lg_bal.x * H * B;
      int sp = 1;
      while (sp < 16 && lg_grid * (sp * 2) <= target) sp *= 2;
      int nblk_cap = (S + 63) / 64;
      while (sp > nblk_cap) sp >>= 1;
      return sp;
    };
    const int sp_leg = auto_split(256), sp_cfg = auto_split(1024);
    auto bench_lse = [&](bool cfg6, int sp, float* out) {
      auto go = [&]() {
        if (cfg6)
          launch_lse_bal<512, 1, false, 128, 16>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv,
                                                 scale, nullptr, d_lse_part, sp);
        else
          launch_lse_bal<512, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, nullptr,
                                 d_lse_part, sp);
      };
      for (int i = 0; i < 3; ++i) go();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) go();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float t_leg = 0.f, t_cfg = 0.f;
    std::vector<float> l_leg((size_t)B * S * H), l_cfg((size_t)B * S * H);
    bench_lse(false, sp_leg, &t_leg);
    CUDA_CHECK(cudaMemcpy(l_leg.data(), d_lse, l_leg.size() * 4, cudaMemcpyDeviceToHost));
    bench_lse(true, sp_cfg, &t_cfg);
    CUDA_CHECK(cudaMemcpy(l_cfg.data(), d_lse, l_cfg.size() * 4, cudaMemcpyDeviceToHost));
    double e = 0.0;
    for (size_t i = 0; i < l_leg.size(); ++i)
      e = std::max(e, (double)std::fabs((double)l_leg[i] - (double)l_cfg[i]));
    printf("[O59 A/B] LSE MLA legacy(sp=%d) %.4f ms | cfg6(sp=%d) %.4f ms (%.3fx) | "
           "max_abs(cfg6-vs-legacy)=%.3e\n",
           sp_leg, t_leg, sp_cfg, t_cfg, t_leg / t_cfg, e);
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

  // ---- O47 A/B（D=512/MLA）：主 kernel 4-warp(128 线程) vs 8-warp(256 线程) ----
  //   只改 warp 网格/累加器划分，数学口径不变；跨 CTA atomic 次序变 ⇒ 不逐位（预期 ~1e-5）。
  if (D == 512) {
    auto go_mla = [&](bool w8) {
      if (w8)
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      else
        launch_bwd_main<512, 64, 32, false, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
    };
    auto zero_acc = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    };
    auto bench_mla = [&](bool w8, float* out) {
      zero_acc();
      go_mla(w8);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) go_mla(w8);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float m4 = 0.f, m8 = 0.f;
    bench_mla(false, &m4);
    bench_mla(true, &m8);
    std::vector<float> dq4(nq), dq8(nq), dk4(nkv), dk8(nkv), dv4(nkv), dv8(nkv);
    auto grab = [&](bool w8, std::vector<float>& dq, std::vector<float>& dk,
                    std::vector<float>& dv) {
      zero_acc();
      go_mla(w8);
      CUDA_CHECK(cudaMemcpy(dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    };
    grab(false, dq4, dk4, dv4);
    grab(true, dq8, dk8, dv8);
    double dqd = 0, dkd = 0, dvd = 0;
    for (size_t i = 0; i < dq4.size(); ++i) {
      dqd = std::max(dqd, (double)std::fabs((double)dq4[i] - (double)dq8[i]));
      dkd = std::max(dkd, (double)std::fabs((double)dk4[i] - (double)dk8[i]));
      dvd = std::max(dvd, (double)std::fabs((double)dv4[i] - (double)dv8[i]));
    }
    printf("[O47 A/B] main MLA 4-warp %.4f ms | 8-warp %.4f ms (%.3fx) | "
           "max_abs(8w-vs-4w) dq=%.3e dk=%.3e dv=%.3e\n",
           m4, m8, m4 / m8, dqd, dkd, dvd);
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O51 A/B（D=512/MLA）：8-warp 主 kernel 的 K/V 同步载入 vs cp.async 回填流水 ----
  //   只改搬运（行主序 cp.async + 从 Ks 重建 Kp），数学/K/V 字节完全相同 ⇒ 应逐位相同。
  if (D == 512) {
    auto go51 = [&](bool kvp) {
      if (kvp)
        launch_bwd_main_kvpipe<512, 64, 32, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      else
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
    };
    auto zero_acc = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    };
    auto bench51 = [&](bool kvp, float* out) {
      zero_acc();
      go51(kvp);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) go51(kvp);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float s_sync = 0.f, s_kvp = 0.f;
    bench51(false, &s_sync);
    bench51(true, &s_kvp);
    std::vector<float> a_dq(nq), b_dq(nq), a_dk(nkv), b_dk(nkv), a_dv(nkv), b_dv(nkv);
    auto grab51 = [&](bool kvp, std::vector<float>& dq, std::vector<float>& dk,
                      std::vector<float>& dv) {
      zero_acc();
      go51(kvp);
      CUDA_CHECK(cudaMemcpy(dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    };
    grab51(false, a_dq, a_dk, a_dv);
    grab51(true, b_dq, b_dk, b_dv);
    auto md51 = [](const std::vector<float>& a, const std::vector<float>& b) {
      double d = 0;
      for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, (double)std::fabs((double)a[i] - (double)b[i]));
      return d;
    };
    printf("[O51 A/B] main MLA 8w sync %.4f ms | kvpipe %.4f ms (%.3fx) | "
           "max_abs(kvp-vs-sync) dq=%.3e dk=%.3e dv=%.3e\n",
           s_sync, s_kvp, s_sync / s_kvp, md51(a_dq, b_dq), md51(a_dk, b_dk), md51(a_dv, b_dv));
    run_main();  // 恢复 CLI 选中路径
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
          else if (D == 256)
            delta_warp_kernel<256><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
          else
            delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        } else {
          if (D == 128) delta_kernel<128><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
          else if (D == 256) delta_kernel<256><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
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
    else if (D == 256) delta_kernel<256><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    else delta_kernel<512><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    std::vector<float> d_old(rows_q);
    CUDA_CHECK(cudaMemcpy(d_old.data(), d_delta, rows_q * 4, cudaMemcpyDeviceToHost));
    if (D == 128)
      delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    else if (D == 256)
      delta_warp_kernel<256><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
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

#ifdef FA_WGMMA
  // ---- F6 A/B（D=128）：wg2（BM=128, 2 wg, mma GEMM1/2）vs wgmma2（GEMM1/2 走 wgmma
  //      + SW128 直读，GEMM3/4/5 仍 mma）。只改 GEMM1/2 的计算通路，数学口径不变。----
  if (D == 128) {
    auto run_wg2b = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_wg2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                          d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                          ksplit2);
    };
    auto run_wg2wg = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_wgmma2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                             d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                             ksplit2);
    };
    auto bench_sel2 = [&](auto&& fn, float* out) {
      fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float t_wg2b = 0.f, t_wg2wg = 0.f;
    bench_sel2(run_wg2b, &t_wg2b);
    bench_sel2(run_wg2wg, &t_wg2wg);
    std::vector<float> a_dq(nq), a_dk(nkv), a_dv(nkv), b_dq(nq), b_dk(nkv), b_dv(nkv);
    run_wg2b();
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_wg2wg();
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd2 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[F6 A/B] main wg2(mma GEMM1/2) %.4f ms | wg2wgmma(GEMM1/2 wgmma) %.4f ms (%.3fx) | "
           "max_abs(wg2wg-vs-wg2) dq/dk/dv=%.3e/%.3e/%.3e\n",
           t_wg2b, t_wg2wg, t_wg2b / t_wg2wg, maxd2(b_dq, a_dq), maxd2(b_dk, a_dk),
           maxd2(b_dv, a_dv));
#if defined(FA_WGMMA) && defined(FA_TMA)
    // O90（F6-step3）：wgmma2 + K/V 4D-TMA 同 session A/B（vs wgmma2，只改 K/V 搬运）。
    auto run_wg2tma = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_wgmma2tma<128>(mg2, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                Hkv, scale, (int)causal, ksplit2);
    };
    float t_wg2tma = 0.f;
    bench_sel2(run_wg2tma, &t_wg2tma);
    std::vector<float> c_dq(nq), c_dk(nkv), c_dv(nkv);
    run_wg2tma();
    CUDA_CHECK(cudaMemcpy(c_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[O90 A/B] main wg2wgmma %.4f ms | wg2wgmma+KVTMA %.4f ms (%.3fx) | "
           "max_abs(KVTMA-vs-wgmma2) dq/dk/dv=%.3e/%.3e/%.3e\n",
           t_wg2wg, t_wg2tma, t_wg2wg / t_wg2tma, maxd2(c_dq, b_dq), maxd2(c_dk, b_dk),
           maxd2(c_dv, b_dv));
#endif
    // O91（F6-step4）：wgmma2（BM=128, 2WG）vs wg3（BM=192, 3WG, 384 线程）同 session A/B。
    //   只改几何/归约结构（BM 64→192 由泛化的 NWG 核承接），数学口径不变，差异应只有 fp32 归约次序。
    auto run_wg3 = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_wgmma_nw<128, 192, 32, 3>(mg3, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                           d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc,
                                           S, H, Hkv, scale, (int)causal, ksplit3);
    };
    float t_wg3 = 0.f;
    bench_sel2(run_wg3, &t_wg3);
    std::vector<float> d_dq(nq), d_dk(nkv), d_dv(nkv);
    run_wg3();
    CUDA_CHECK(cudaMemcpy(d_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[O91 A/B] main wgmma2(BM128,grid=%d) %.4f ms | wg3(BM192,grid=%d,ksplit3=%d) "
           "%.4f ms (%.3fx) | max_abs(wg3-vs-wgmma2) dq/dk/dv=%.3e/%.3e/%.3e\n",
           mg2.x, t_wg2wg, mg3.x, ksplit3, t_wg3, t_wg2wg / t_wg3, maxd2(d_dq, b_dq),
           maxd2(d_dk, b_dk), maxd2(d_dv, b_dv));
    run_main();  // 恢复 CLI 选中路径
  }
#endif

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

  // P3-3：落盘 ours 输出（默认关，行为逐位不变）。
  if (!dump_prefix.empty()) {
    fa_bwd_save_npy_f32(dir + "/" + dump_prefix + "_dq.npy", mdq.data(), (long long)mdq.size());
    fa_bwd_save_npy_f32(dir + "/" + dump_prefix + "_dk.npy", mdk.data(), (long long)mdk.size());
    fa_bwd_save_npy_f32(dir + "/" + dump_prefix + "_dv.npy", mdv.data(), (long long)mdv.size());
    printf("[dump] ours -> %s/%s_{dq,dk,dv}.npy\n", dir.c_str(), dump_prefix.c_str());
  }

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
