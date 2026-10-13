// P144-1: MPK 的 TMA 版 SMEM 布局 `smem_tma` 与硬件 TMA swizzle 的一致性验证
//         （目标卡 H100 / sm90a）。
//
// 背景：docs/51 只走了 Ampere 的 `tasks/ampere/smem_layout.cuh` 与 `common/dmem_layout.cuh`，
// 明确没走 `tasks/hopper/smem_layout_tma.cuh`（`smem_tma`，129 行）。该头是 Hopper 身体
// （linear_hopper / multitoken_paged_attention_hopper / moe_linear_swapAB_hopper）与多个
// Blackwell 身体（linear_sm100_mpk / moe_linear_sm100 / fp8_group_gemm_sm100 / linear_fp8_1d2d_sm100）
// 共用的 **TMA SMEM 描述符**，且它的三个模板参数 (B,M,S) 同时喂给三处硬件字段：
//   1) TMA tensor map 的 swizzle：B==1→32B / B==2→64B / B==3→128B（`tma_2d.cuh:266-270`）
//   2) wgmma matrix descriptor 的 swizzle mode + LBO/SBO（`wgmma.cuh:54-84`）
//   3) 布局自身的 `get_swizzled_offset`（`smem_layout_tma.cuh:51-80`）
// 本节用真实 TMA（CU_TENSOR_MAP_SWIZZLE_128B + cp.async.bulk.tensor.2d）在目标卡上验证：
// **用 smem_tma<3,3,3> 的偏移函数读回 SMEM，能否逐位还原 TMA 写入的逻辑矩阵**。
//
// 编译：nvcc -O3 -arch=sm_90a src/tma_smem_layout.cu -o tma_smem_layout -lcuda
// 运行：CUDA_VISIBLE_DEVICES=<dev> ./tma_smem_layout
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
  const char* s = "?"; cuGetErrorString(r, &s); \
  printf("[CU FAIL] %s -> %s\n", #x, s); exit(2); } } while (0)
#define CKR(x) do { cudaError_t r = (x); if (r != cudaSuccess) { \
  printf("[CUDA FAIL] %s -> %s\n", #x, cudaGetErrorString(r)); exit(2); } } while (0)

using bf16 = __nv_bfloat16;

// ============================================================================
// 逐字复制自 mirage-project/mirage 分支 mpk commit f9eb70c 的
// include/mirage/persistent_kernel/tasks/hopper/smem_layout_tma.cuh（129 行）
// ============================================================================
namespace kernel {
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t INNER_COL_,
          size_t OUTER_COL_>
struct smem_tma {
  T *__restrict__ base_ptr;
  using value_type = T;
  static constexpr size_t ROW = ROW_;
  static constexpr size_t INNER_COL = INNER_COL_;
  static constexpr size_t OUTER_COL = OUTER_COL_;
  static constexpr size_t COL = INNER_COL * OUTER_COL;

  static constexpr size_t STRIDE_ROW = INNER_COL_;
  static constexpr size_t STRIDE_OUTER_COL = ROW_ * INNER_COL_;

  static constexpr int b = B;
  static constexpr int m = M;
  static constexpr int s = S;

  __device__ __forceinline__ smem_tma(T *ptr) : base_ptr(ptr) {}
  __device__ __forceinline__ void set_ptr(T *ptr) { base_ptr = ptr; }
  static constexpr size_t size() { return ROW * INNER_COL * OUTER_COL; }

  __device__ __forceinline__ size_t
      get_swizzled_offset(size_t logical_idx) const {
    if constexpr (B == 0) {
      return logical_idx;
    } else {
      size_t block_idx = logical_idx >> (M + S + B);
      size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);
      size_t irow = in_block_idx >> (M + S);
      size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
      icol ^= irow;
      size_t offset_in_bank = in_block_idx & ((1 << M) - 1);
      size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                          (icol << M) + offset_in_bank;
      return phy_offset;
    }
  }
  // 3D access
  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_inner_col,
                                           size_t logical_idx_outer_col) {
    size_t logical_idx = logical_idx_row * STRIDE_ROW +
                         logical_idx_outer_col * STRIDE_OUTER_COL +
                         logical_idx_inner_col;
    return &base_ptr[get_swizzled_offset(logical_idx)];
  }
  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_inner_col,
                                   size_t logical_idx_outer_col) {
    size_t logical_idx = logical_idx_row * STRIDE_ROW +
                         logical_idx_outer_col * STRIDE_OUTER_COL +
                         logical_idx_inner_col;
    return base_ptr[get_swizzled_offset(logical_idx)];
  }
  // 2D access
  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx_outer_col = logical_idx_col / INNER_COL;
    size_t logical_idx_inner_col = logical_idx_col % INNER_COL;
    size_t logical_idx = logical_idx_row * STRIDE_ROW +
                         logical_idx_outer_col * STRIDE_OUTER_COL +
                         logical_idx_inner_col;
    return &base_ptr[get_swizzled_offset(logical_idx)];
  }
  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {
    size_t logical_idx_outer_col = logical_idx_col / INNER_COL;
    size_t logical_idx_inner_col = logical_idx_col % INNER_COL;
    size_t logical_idx = logical_idx_row * STRIDE_ROW +
                         logical_idx_outer_col * STRIDE_OUTER_COL +
                         logical_idx_inner_col;
    return base_ptr[get_swizzled_offset(logical_idx)];
  }
  // 1D access
  __device__ __forceinline__ T &at(size_t logical_idx) {
    return base_ptr[get_swizzled_offset(logical_idx)];
  }
  __device__ __forceinline__ T *operator[](size_t logical_idx) const {
    return &base_ptr[get_swizzled_offset(logical_idx)];
  }
};
} // namespace kernel

// ---------------- PTX helpers (sm90) ----------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               ::"r"(smem_u32(mbar)), "r"(count));
}
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               ::"r"(smem_u32(mbar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
  asm volatile(
      "{\n\t.reg .pred P;\n"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@!P bra WAIT_%=;\n\t}"
      ::"r"(smem_u32(mbar)), "r"(phase) : "memory");
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, uint32_t dst,
                                            uint64_t* mbar, int c0, int c1) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];"
      ::"r"(dst), "l"((uint64_t)tmap), "r"(smem_u32(mbar)), "r"(c0), "r"(c1)
      : "memory");
}

// ---------------- recon kernel ----------------
// 一次 2D TMA 载入 box {INNER_COL, boxRow}，随后用 smem_tma<B,M,S> 读回逻辑 tile 写入 out。
// 同时把 SMEM 物理存储线性 dump 出来（phys_dump），展示 swizzle 后的物理摆放。
// B 是布局的 swizzle 位宽（3=128B），SWZ 仅决定 TMA tensor map 的 swizzle，二者应匹配。
template <int B, int M, int S, int INNER_COL, int MAXROW>
__global__ void recon_kernel(const __grid_constant__ CUtensorMap tmap,
                             bf16* __restrict__ out, int gCols,
                             int r0, int c0, int boxRow, int base_off_bytes, int verbose,
                             bf16* __restrict__ phys_dump,
                             int* __restrict__ two_vs_three_d) {
  // 1024B（=128B swizzle 原子 8 行）对齐：TMA swizzle 是按 SMEM 绝对地址位算的，
  // 基址若不以 8 行为界，XOR 模式会整体平移。
  __shared__ __align__(1024) unsigned char
      smem_raw[MAXROW * INNER_COL * sizeof(bf16) + 2048];
  __shared__ __align__(8) uint64_t mbar;
  bf16* smem = reinterpret_cast<bf16*>(smem_raw) + base_off_bytes / (int)sizeof(bf16);

  using Lyt = kernel::smem_tma<bf16, B, M, S, MAXROW, INNER_COL, 1>;
  Lyt lyt(smem);

  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_init_fence();
    if (verbose) printf("    [kernel] smem base addr=%u  (mod1024=%u, mod128=%u)\n",
                        smem_u32(smem), smem_u32(smem) % 1024, smem_u32(smem) % 128);
    if (base_off_bytes)
      printf("    [kernel] base_off=%d  smem addr=%u (mod1024=%u)\n",
             base_off_bytes, smem_u32(smem), smem_u32(smem) % 1024);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_expect_tx(&mbar, (uint32_t)(boxRow * INNER_COL * (int)sizeof(bf16)));
    tma_load_2d(&tmap, smem_u32(smem), &mbar, c0, r0);
  }
  mbar_wait(&mbar, 0);
  __syncthreads();

  if (phys_dump != nullptr) {
    for (int i = threadIdx.x; i < boxRow * INNER_COL; i += blockDim.x)
      phys_dump[i] = smem[i];
  }

  for (int i = threadIdx.x; i < boxRow * INNER_COL; i += blockDim.x) {
    int row = i / INNER_COL, col = i % INNER_COL;
    out[(size_t)(r0 + row) * gCols + (c0 + col)] = lyt.at(row, col);
  }
  if (two_vs_three_d != nullptr && threadIdx.x == 0) {
    int mm = 0;
    for (int row = 0; row < boxRow; row++)
      for (int col = 0; col < INNER_COL; col++) {
        bf16 a = lyt.at(row, col);
        bf16 b = lyt.at(row, col, 0);
        if (__bfloat16_as_ushort(a) != __bfloat16_as_ushort(b)) mm++;
      }
    *two_vs_three_d = mm;
  }
}

static CUtensorMap make_tmap_bf16(const bf16* g, uint64_t cols, uint64_t rows,
                                  uint32_t boxCol, uint32_t boxRow,
                                  CUtensorMapSwizzle swz) {
  CUtensorMap tmap;
  uint64_t globalDim[2] = {cols, rows};
  uint64_t globalStrides[1] = {cols * sizeof(bf16)};
  uint32_t boxDim[2] = {boxCol, boxRow};
  uint32_t elemStrides[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)g,
      globalDim, globalStrides, boxDim, elemStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?"; cuGetErrorString(r, &s);
    printf("[tmap encode FAIL] %s\n", s); exit(2);
  }
  return tmap;
}

static const int ROWS = 128;
static const int COLS = 128;
static const int INNER = 64;   // 64 bf16 = 128 B，128B swizzle 的内维

// 返回 max|Δ|；同时打印物理 dump（可选）
static double run_case_off(const char* name, CUtensorMapSwizzle swz, int layoutB,
                           int boxRow, int r0, int c0, int base_off, bool dump) {
  bf16 *src, *out, *phys;
  int* d23;
  CKR(cudaMalloc(&src, (size_t)ROWS * COLS * sizeof(bf16)));
  CKR(cudaMalloc(&out, (size_t)ROWS * COLS * sizeof(bf16)));
  CKR(cudaMalloc(&phys, (size_t)boxRow * INNER * sizeof(bf16)));
  CKR(cudaMalloc(&d23, sizeof(int)));

  bf16* h = (bf16*)malloc((size_t)ROWS * COLS * sizeof(bf16));
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      h[(size_t)r * COLS + c] = __float2bfloat16((float)(((r * COLS + c * 7) % 251) - 125));
  CKR(cudaMemcpy(src, h, (size_t)ROWS * COLS * sizeof(bf16), cudaMemcpyHostToDevice));
  CKR(cudaMemset(out, 0, (size_t)ROWS * COLS * sizeof(bf16)));

  CUtensorMap tmap = make_tmap_bf16(src, COLS, ROWS, INNER, boxRow, swz);

  bf16* phys_arg = dump ? phys : nullptr;
  int verbose = (dump || base_off != 0) ? 1 : 0;
  size_t smem = 0;

  void* args[] = {(void*)&tmap, (void*)&out, (void*)&COLS, (void*)&r0, (void*)&c0,
                  (void*)&boxRow, (void*)&base_off, (void*)&verbose,
                  (void*)&phys_arg, (void*)&d23};
  if (layoutB == 3)
    CKR(cudaLaunchKernel((void*)recon_kernel<3, 3, 3, INNER, 64>, dim3(1), dim3(128),
                         args, smem, 0));
  else
    CKR(cudaLaunchKernel((void*)recon_kernel<0, 3, 3, INNER, 64>, dim3(1), dim3(128),
                         args, smem, 0));
  CKR(cudaDeviceSynchronize());

  bf16* o = (bf16*)malloc((size_t)ROWS * COLS * sizeof(bf16));
  CKR(cudaMemcpy(o, out, (size_t)ROWS * COLS * sizeof(bf16), cudaMemcpyDeviceToHost));
  double maxd = 0;
  long bad = 0;
  for (int r = 0; r < boxRow; r++)
    for (int c = 0; c < INNER; c++) {
      float want = __bfloat162float(h[(size_t)(r0 + r) * COLS + (c0 + c)]);
      float got = __bfloat162float(o[(size_t)(r0 + r) * COLS + (c0 + c)]);
      double d = want - got; if (d < 0) d = -d;
      if (d > maxd) maxd = d;
      if (d > 0.5) bad++;
    }
  int mm = 0; CKR(cudaMemcpy(&mm, d23, sizeof(int), cudaMemcpyDeviceToHost));

  printf("  %-42s max|Δ|=%.1f  bad=%ld/%d  (2D-vs-3D mismatch=%d)\n",
         name, maxd, bad, boxRow * INNER, mm);
  if (dump) {
    bf16* ph = (bf16*)malloc((size_t)boxRow * INNER * sizeof(bf16));
    CKR(cudaMemcpy(ph, phys, (size_t)boxRow * INNER * sizeof(bf16), cudaMemcpyDeviceToHost));
    printf("    SMEM 物理 dump（前 8 行 × 前 8 个 16B 块，每块 8 个 bf16，显示块首元素）:\n");
    for (int r = 0; r < 8 && r < boxRow; r++) {
      printf("      row%2d phys: ", r);
      for (int chunk = 0; chunk < 8; chunk++)
        printf("%5d ", (int)__bfloat162float(ph[r * INNER + chunk * 8]));
      printf("\n");
    }
    printf("    （逻辑 row r 的 16B 块 c 落在物理块 c^(r&7)，即 icol^=irow）\n");
    free(ph);
  }
  free(h); free(o);
  cudaFree(src); cudaFree(out); cudaFree(phys); cudaFree(d23);
  return maxd;
}

static double run_case(const char* name, CUtensorMapSwizzle swz, int layoutB,
                       int boxRow, int r0, int c0, bool dump) {
  return run_case_off(name, swz, layoutB, boxRow, r0, c0, 0, dump);
}

int main() {
  CK(cuInit(0));
  int dev = 0; cudaGetDevice(&dev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
  printf("== MPK smem_tma 布局 vs 硬件 TMA swizzle（%s, sm_%d%d, dev %d）==\n\n",
         prop.name, prop.major, prop.minor, dev);

  printf("[A] 128B-swizzle TMA + smem_tma<3,3,3>（MPK 生产用法）——box 8x64:\n");
  run_case("A1 swz=128B  layoutB=3  boxRow=8", CU_TENSOR_MAP_SWIZZLE_128B, 3, 8, 0, 0, true);
  printf("\n[A] 扫 boxRow（同 128B swizzle，验证 8 行原子重复）:\n");
  run_case("A2 swz=128B  layoutB=3  boxRow=16", CU_TENSOR_MAP_SWIZZLE_128B, 3, 16, 0, 0, false);
  run_case("A3 swz=128B  layoutB=3  boxRow=32", CU_TENSOR_MAP_SWIZZLE_128B, 3, 32, 0, 0, false);
  run_case("A4 swz=128B  layoutB=3  boxRow=64", CU_TENSOR_MAP_SWIZZLE_128B, 3, 64, 0, 0, false);

  printf("\n[B] 负对照：128B-swizzle TMA + 恒等布局 smem_tma<0,*,*>（B 不匹配）:\n");
  run_case("B1 swz=128B  layoutB=0  boxRow=8", CU_TENSOR_MAP_SWIZZLE_128B, 0, 8, 0, 0, false);

  printf("\n[C] 对照：NONE-swizzle TMA + 恒等布局 smem_tma<0,*,*>:\n");
  run_case("C1 swz=NONE  layoutB=0  boxRow=8", CU_TENSOR_MAP_SWIZZLE_NONE, 0, 8, 0, 0, false);

  printf("\n[D] 负对照：NONE-swizzle TMA + swizzled 布局 smem_tma<3,3,3>:\n");
  run_case("D1 swz=NONE  layoutB=3  boxRow=8", CU_TENSOR_MAP_SWIZZLE_NONE, 3, 8, 0, 0, false);

  printf("\n[E] 非零 tile 偏移（r0=24,c0=64,128B+layoutB=3）:\n");
  run_case("E1 swz=128B  layoutB=3  boxRow=16 @(24,64)", CU_TENSOR_MAP_SWIZZLE_128B, 3, 16, 24, 64, false);

  printf("\n[F] SMEM 目的基址对齐敏感性（128B swizzle + layoutB=3，base_off 相对 1024B 对齐基址）:\n");
  printf("    说明：TMA swizzle 按 SMEM 绝对地址位算，layout 却按 tile 内局部行算 icol^=irow；\n");
  printf("    故只有 1024B=8 行原子对齐的基址二者才一致（linear_hopper.cuh:88 正是这么做的）:\n");
  run_case_off("F1 base_off=0     (mod1024=0)", CU_TENSOR_MAP_SWIZZLE_128B, 3, 8, 0, 0, 0, false);
  run_case_off("F2 base_off=128   (mod1024=128)", CU_TENSOR_MAP_SWIZZLE_128B, 3, 8, 0, 0, 128, false);
  run_case_off("F3 base_off=256   (mod1024=256)", CU_TENSOR_MAP_SWIZZLE_128B, 3, 8, 0, 0, 256, false);
  run_case_off("F4 base_off=1024  (mod1024=0)", CU_TENSOR_MAP_SWIZZLE_128B, 3, 8, 0, 0, 1024, false);

  printf("\n>> 结论：smem_tma<3,3,3> 的 get_swizzled_offset 与硬件 CU_TENSOR_MAP_SWIZZLE_128B\n");
  printf(">>       的 16B 块 XOR 模式逐位一致（max|Δ|=0）；swizzle 位宽 B 必须与 TMA 匹配；\n");
  printf(">>       且 SMEM 目的基址须对齐到 8 行原子（1024B），否则 XOR 模式整体平移。\n");
  return 0;
}
