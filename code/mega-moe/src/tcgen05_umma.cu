// P100-1 SM100 UMMA（tcgen05）指令底座 + SM100 `bmk,bnk->mn` 批归约 GEMM 的 H100 可跑部分：
//   deep_gemm/include/deep_gemm/ptx/tcgen05.cuh           （UMMA/tcgen05 指令封装层）
//   deep_gemm/include/deep_gemm/impls/sm100_bmk_bnk_mn.cuh（本层最简单的消费者）
//   csrc/jit_kernels/impls/sm100_bmk_bnk_mn.hpp            （host 包装层）
//
// 本文件在 H100(sm90a) 真机编译运行——只跑**架构无关**的部分（真实的 tcgen05 指令本身是 SM100 专属，
//   其在 sm_90a 下被 ptxas 拒绝的“硬证据”由 harness 的编译矩阵给出，见 *.sh 的 Part A）。
//   Part 1  host：STSM swizzle 整数公式 vs cute::Swizzle<3,4,3>（bmk 内层 epilogue 用的映射）
//   Part 2  device：真实 TMA reduce-add（cp.reduce.async.bulk.tensor.2d ... add.f32）累积语义
//   Part 3  device：swizzle-128 的 TMA reduce-add（对齐 bmk kernel D 描述符 + epilogue 写回）
//   Part 4  host：真实 ptx/tcgen05.cuh 在 sm90a 下可解析；UMMA 指令描述符/尺寸常量静态校验
//
// 编译：见 harness/exp_tcgen05_umma.sh
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/atom/copy_traits_sm100.hpp>   // 把 cute::SM100_TMEM_LOAD_* 带进 namespace cute
#include <cutlass/bfloat16.h>

// 真实指令底座（含 9 个 tcgen05.mma 封装 / tmem_load / umma_arrive / 两条 fence）
#include <deep_gemm/ptx/tcgen05.cuh>
// 描述符构造层（bmk kernel 用 SM100_MMA_F16BF16_SS 消费这里的 desc）
#include <deep_gemm/mma/sm100.cuh>

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& tag) {
    if (ok) { g_pass++; } else { g_fail++; printf("  !! FAIL %s\n", tag.c_str()); }
}
#define CU_CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
    printf("  !! CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); g_fail++; } } while (0)

// ---------------------------------------------------------------------------
// tensormap 构造（直连 driver；等价 runtime_utils.hpp 的 make_tma_2d_desc）
// ---------------------------------------------------------------------------
static CUtensorMap make_map_2d_f32(void* ptr, uint64_t inner, uint64_t outer,
                                   uint32_t box_inner, uint32_t box_outer, CUtensorMapSwizzle swz) {
    CUtensorMap m;
    const uint64_t dims[2] = {inner, outer};
    const uint64_t strides[1] = {inner * sizeof(float)};
    const uint32_t box[2] = {box_inner, box_outer};
    const uint32_t es[2] = {1, 1};
    const CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, ptr, dims, strides,
        box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { printf("  !! cuTensorMapEncodeTiled failed %d\n", (int)r); g_fail++; }
    return m;
}

// ---------------------------------------------------------------------------
// Part 1：STSM swizzle 整数公式 vs cute::Swizzle<3,4,3>
//   取自 sm100_bmk_bnk_mn.cuh:223-249（kSwizzleCDMode=128 / float）逐行原样抽取。
// ---------------------------------------------------------------------------
// 返回：在 [BLOCK_M=128, STORE_BLOCK_N=32] fp32 子块里，(global_row, col8) 对应的 **smem 字节偏移**。
//   - col8 = 16B bank group 索引（0..7），within = 16B 组内 float 索引（0..3）
//   - 公式每 warp 覆盖 32 行（base=warp*32*128B），原子内 row = lane（0..31）
static uint32_t bmk_stsm_byte_offset(const uint32_t global_row, const uint32_t col8) {
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kSwizzleCDMode = 128;
    const uint32_t warp_idx = global_row / 32;
    const uint32_t lane_idx = global_row % 32;
    // kHasShortcut = (kSwizzleCDMode / kNumBankGroupBytes) == 8 -> true
    const uint32_t row = lane_idx;              // = i/8 + lane_idx with i in [0,8)
    uint32_t col = col8;
    col ^= row % (kSwizzleCDMode / 16);         // row % 8
    auto* smem_base = reinterpret_cast<uint8_t*>(0) + warp_idx * 32 * kSwizzleCDMode;
    auto* smem_ptr = smem_base + row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
    return static_cast<uint32_t>(smem_ptr - reinterpret_cast<uint8_t*>(0));
}

static void part1_swizzle_listing() {
    printf("== Part 1: STSM swizzle 公式 vs cute::Swizzle<3,4,3> ==\n");
    cute::Swizzle<3, 4, 3> sw;
    uint32_t mism = 0, total = 0;
    for (uint32_t r = 0; r < 128; ++r) {
        for (uint32_t c8 = 0; c8 < 8; ++c8) {
            const uint32_t got = bmk_stsm_byte_offset(r, c8);
            const uint32_t linear = r * 128 + c8 * 16;
            const uint32_t want = static_cast<uint32_t>(sw(linear));
            if (got != want) { if (mism < 5) printf("    r=%u c=%u got=%u want=%u\n", r, c8, got, want); mism++; }
            total++;
        }
    }
    printf("  (row,col8) 双射检查 mismatch=%u/%u\n", mism, total);
    check(mism == 0, "Part1 bmk swizzle == Swizzle<3,4,3>");
    // 单射（同一 smem 地址不被两个逻辑元素占用）
    std::vector<char> seen(128 * 128, 0);
    uint32_t coll = 0;
    for (uint32_t r = 0; r < 128; ++r) for (uint32_t c8 = 0; c8 < 8; ++c8) {
        const uint32_t b = bmk_stsm_byte_offset(r, c8);
        for (uint32_t w = 0; w < 16; ++w) { if (seen[b + w]) coll++; seen[b + w] = 1; }
    }
    printf("  16B 组内展开后碰撞=%u/16384\n", coll);
    check(coll == 0, "Part1 16B chunk bijection");
}

// ---------------------------------------------------------------------------
// Part 2/3：TMA reduce-add 设备内核
// ---------------------------------------------------------------------------
template <uint32_t BM, uint32_t BOX_N, uint32_t STRIDE_N>
__global__ void reduce_add_plain_kernel(const __grid_constant__ CUtensorMap dmap, const float fill) {
    extern __shared__ __align__(1024) float smem[];
    for (int i = threadIdx.x; i < (int)(BM * BOX_N); i += blockDim.x) smem[i] = fill;
    __syncthreads();
    if (threadIdx.x == 0) {
        cute::SM90_TMA_REDUCE_ADD_2D::copy(&dmap, smem, 0, 0);
        cute::tma_store_arrive();
    }
    __syncthreads();
    cute::tma_store_wait<0>();
}

// swizzle-128：smem 用 cute::Swizzle<3,4,3> 铺一个 [BM x 32] fp32 tile，再 reduce-add 到 D 的第 col0 列
template <uint32_t BM>
__global__ void reduce_add_swizzle_kernel(const __grid_constant__ CUtensorMap dmap,
                                          const float fill, const uint32_t col0) {
    extern __shared__ __align__(1024) float smem[];
    cute::Swizzle<3, 4, 3> sw;
    for (int idx = threadIdx.x; idx < (int)(BM * 32); idx += blockDim.x) {
        const uint32_t r = idx / 32, c = idx % 32;
        // 逻辑 (r,c) -> swizzled 字节偏移（float 元素）
        const uint32_t boff = static_cast<uint32_t>(sw(r * 128 + c * 4));
        *reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem) + boff) = fill;
    }
    __syncthreads();
    cute::tma_store_fence();
    __syncthreads();
    if (threadIdx.x == 0) {
        cute::SM90_TMA_REDUCE_ADD_2D::copy(&dmap, smem, (int32_t)col0, 0);
        cute::tma_store_arrive();
    }
    __syncthreads();
    cute::tma_store_wait<0>();
}

static void part2_reduce_add_plain() {
    printf("== Part 2: 真实 TMA reduce-add 累积语义（plain, 128x128 fp32）==\n");
    const uint32_t M = 128, N = 128;
    float* d = nullptr; CU_CHECK(cudaMalloc(&d, M * N * sizeof(float)));
    auto dmap = make_map_2d_f32(d, N, M, N, M, CU_TENSOR_MAP_SWIZZLE_NONE);
    // D 初值 0.5
    { std::vector<float> host(M * N, 0.5f); CU_CHECK(cudaMemcpy(d, host.data(), M * N * 4, cudaMemcpyHostToDevice)); }
    // 64KB 动态 smem 需抬高上限（否则 launch 静默 invalid argument）
    CU_CHECK(cudaFuncSetAttribute(reduce_add_plain_kernel<128, 128, 128>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, M * N * sizeof(float)));
    auto launch = [&](float fill) {
        reduce_add_plain_kernel<128, 128, 128><<<1, 128, M * N * sizeof(float)>>>(dmap, fill);
        CU_CHECK(cudaGetLastError());
        CU_CHECK(cudaDeviceSynchronize());
    };
    launch(1.0f); launch(1.0f); launch(1.0f);
    std::vector<float> host(M * N, 0.f); CU_CHECK(cudaMemcpy(host.data(), d, M * N * 4, cudaMemcpyDeviceToHost));
    float mn = 1e30f, mx = -1e30f; for (float v : host) { mn = fminf(mn, v); mx = fmaxf(mx, v); }
    printf("  0.5 + 3x(1.0) -> min=%.4f max=%.4f (期望 3.5)\n", mn, mx);
    check(mn == 3.5f && mx == 3.5f, "Part2 add accumulates");
    // 反例：若 kernel 是“覆盖”而非 reduce-add，结果会是 1.0
    printf("  (反例：若为覆盖写结果应为 1.0)\n");
    CU_CHECK(cudaFree(d));
}

static void part3_reduce_add_swizzle() {
    printf("== Part 3: swizzle-128 TMA reduce-add（对齐 bmk kernel D 描述符 + epilogue）==\n");
    const uint32_t M = 128, N = 128;
    float* d = nullptr; CU_CHECK(cudaMalloc(&d, M * N * sizeof(float)));
    // swizzle-128 -> box inner = 128B/4 = 32 floats
    auto dmap = make_map_2d_f32(d, N, M, /*box_inner*/32, /*box_outer*/M, CU_TENSOR_MAP_SWIZZLE_128B);
    { std::vector<float> host(M * N, 0.f); CU_CHECK(cudaMemcpy(d, host.data(), M * N * 4, cudaMemcpyHostToDevice)); }
    // 两次 reduce-add，分别打到 col0=0 与 col0=32
    reduce_add_swizzle_kernel<128><<<1, 128, M * 32 * sizeof(float)>>>(dmap, 2.0f, 0);
    CU_CHECK(cudaDeviceSynchronize());
    reduce_add_swizzle_kernel<128><<<1, 128, M * 32 * sizeof(float)>>>(dmap, 3.0f, 32);
    CU_CHECK(cudaDeviceSynchronize());
    std::vector<float> host(M * N, 0.f); CU_CHECK(cudaMemcpy(host.data(), d, M * N * 4, cudaMemcpyDeviceToHost));
    uint32_t bad = 0;
    for (uint32_t r = 0; r < M; ++r) for (uint32_t c = 0; c < N; ++c) {
        const float want = (c < 32) ? 2.0f : (c < 64 ? 3.0f : 0.0f);
        if (host[r * N + c] != want) bad++;
    }
    printf("  D[:,0:32]+=2.0, D[:,32:64]+=3.0 -> mismatch=%u/16384\n", bad);
    check(bad == 0, "Part3 swizzled reduce-add orientation ok");
    CU_CHECK(cudaFree(d));
}

// ---------------------------------------------------------------------------
// Part 4：真实 ptx/tcgen05.cuh 在 sm90a 下可解析；常量静态校验
// ---------------------------------------------------------------------------
static void part4_static() {
    printf("== Part 4: tcgen05.cuh 在 sm90a 下可解析 + 描述符常量 ==\n");
    // SmemDescriptor 是 64 位（lo 32 = smem desc，hi 32 = instruction desc）
    check(sizeof(cute::UMMA::SmemDescriptor) == 8, "SmemDescriptor == 8B");
    // cute 指令描述符类型存在（bmk 用其 hi 32 位给 tcgen05.mma）
    printf("  sizeof(InstrDescriptor)=%zu sizeof(InstrDescriptorBlockScaled)=%zu\n",
           sizeof(cute::UMMA::InstrDescriptor), sizeof(cute::UMMA::InstrDescriptorBlockScaled));
    check(sizeof(cute::UMMA::InstrDescriptor) == 4, "InstrDescriptor == 4B");
    check(sizeof(cute::UMMA::InstrDescriptorBlockScaled) == 4, "InstrDescriptorBlockScaled == 4B");
}

int main() {
    printf("### P100-1 tcgen05 UMMA 指令底座 + SM100 bmk_bnk_mn 实验（H100 sm90a）###\n");
    printf("include <deep_gemm/ptx/tcgen05.cuh> 与 <deep_gemm/mma/sm100.cuh> 在本 TU 解析成功\n\n");
    part1_swizzle_listing();
    part2_reduce_add_plain();
    part3_reduce_add_swizzle();
    part4_static();
    printf("\n==== 汇总：PASS=%d FAIL=%d ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
