// P154-1 实验：DeepGEMM SM90 TF32 WGMMA 的 RS（寄存器 A）路径。
//
// 背景：docs/49（P39-1）拆了 SM90 WGMMA 描述符/选择器/异步组，并真机跑了 BF16/FP8 的 SS 路径；
//   但它把「TF32MMASelector 的 RS 路径数值」明确列为未做（docs/49 §10）。本文件补上这一格。
//
// 为什么 H100(sm90a) 可跑：`deep_gemm/include/deep_gemm/mma/sm90.cuh` 的
//   TF32MMARS / TF32MMASelector 与 ptx/wgmma.cuh 的 fence/commit/wait 都是 SM90 设备代码，
//   本机 H100 直接编译运行真实 WGMMA 指令（`wgmma.mma_async ... .f32.tf32.tf32` 的 RS 形态：
//   A 在寄存器、B 在 smem）。
//
// 本文件：
//   (1) 静态属性：TF32MMASelector<N,true> 的 M/N/K/kNumAccum（对照 docs/49 的 BF16/FP8）。
//   (2) A 寄存器布局：用 cute 的 ALayout_64x8 公式把 (lane,reg) 映射到 (m,k)，dump 出来，
//       Python 独立复刻核对（并验证它恰好等于 sm90_tf32_hc_prenorm_gemm.cuh 里手写的取址）。
//   (3) 真实 WGMMA 数值：A 寄存器 + B(smem, 128B swizzle) → D[64,N]，写盘，Python 对 fp64 参考。
//       四组输入分别隔离「A 舍入」「B 舍入」「两者都不舍入」。
//
// 编译：nvcc -gencode=arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr ...
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cuda.h>

#include <cute/numeric/integer_sequence.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>

using namespace deep_gemm;
namespace mma90 = deep_gemm::mma::sm90;

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e) { \
    printf("CUDA error: %s at line %d\n", cudaGetErrorString(_e), __LINE__); exit(1); } } while (0)

static void write_bin(const char* path, const void* p, size_t bytes);

// ---------------------------------------------------------------------------
// (1) 静态属性 dump
// ---------------------------------------------------------------------------
__global__ void static_dump(uint64_t* out) {
    out[0] = mma90::TF32MMASelector<8>::type::M;
    out[1] = mma90::TF32MMASelector<8>::type::K;
    out[2] = mma90::TF32MMASelector<8>::type::kNumAccum;
    out[3] = mma90::TF32MMASelector<64>::type::M;
    out[4] = mma90::TF32MMASelector<64>::type::N;
    out[5] = mma90::TF32MMASelector<64>::type::K;
    out[6] = mma90::TF32MMASelector<64>::type::kNumAccum;
    out[7] = mma90::TF32MMASelector<128>::type::M;
    out[8] = mma90::TF32MMASelector<128>::type::N;
    out[9] = mma90::TF32MMASelector<128>::type::K;
    out[10] = mma90::TF32MMASelector<128>::type::kNumAccum;
    out[11] = mma90::TF32MMASelector<256>::type::kNumAccum;
}

// ---------------------------------------------------------------------------
// (2) A 寄存器布局：ALayout_64x8 (thread,reg) -> (m,k)
//     shape/strides 见 cute/atom/mma_traits_sm90_gmma.hpp：
//     Layout<Shape<Shape<_4,_8,_4>, Shape<_2,_2>>, Stride<Stride<_64,_1,_16>, Stride<_8,_256>>>
// ---------------------------------------------------------------------------
__device__ __forceinline__ void alayout_64x8(int lane, int reg, int& m, int& k) {
    const int t0 = lane % 4, t1 = (lane / 4) % 8, t2 = lane / 32;
    const int v0 = reg % 2, v1 = reg / 2;
    // offset = m + 64*k  （m 为主维，k 为 64 步长）
    m = t1 + 16 * t2 + 8 * v0;
    k = t0 + 4 * v1;
}

__global__ void alayout_dump(int* out) {
    const int lane = threadIdx.x;
    for (int r = 0; r < 4; ++ r) {
        int m, k; alayout_64x8(lane, r, m, k);
        out[(lane * 4 + r) * 2 + 0] = m;
        out[(lane * 4 + r) * 2 + 1] = k;
    }
}

// ---------------------------------------------------------------------------
// (3) 真实 WGMMA：A[64,BLOCK_K]（寄存器，RS） @ B[BLOCK_N,BLOCK_K]^T（smem，128B swizzle）
//     BLOCK_K 必须是 32（=128 字节，满足 K-major 的 kSwizzleMode==BLOCK_K*sizeof(float)）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fill_swizzle_128_float(float* smem, const float* glob, int rows, int k) {
    for (int i = threadIdx.x; i < rows * k; i += blockDim.x) {
        int row = i / k, col = i % k;
        int chunk = col / 4, within = col % 4;   // 16B chunk = 4 个 float
        int chunk2 = chunk ^ (row & 7);          // SWIZZLE_128B
        smem[row * k + chunk2 * 4 + within] = glob[i];
    }
}

template <typename MMA, int BLOCK_N, int BLOCK_K>
__global__ void tf32_rs_gemm(const float* A, const float* B, float* D) {
    constexpr int BLOCK_M = 64;
    static_assert(BLOCK_K == 32, "BLOCK_K must be 32 (=128 bytes)");
    extern __shared__ __align__(1024) uint8_t smem[];
    float* sb = reinterpret_cast<float*>(smem);

    fill_swizzle_128_float(sb, B, BLOCK_N, BLOCK_K);
    __syncthreads();

    const int lane = threadIdx.x;
    constexpr int kNumRegPerWgmma = MMA::M * MMA::K / 128;   // 4
    constexpr int kNumSteps = BLOCK_K / MMA::K;              // 4

    float a[kNumRegPerWgmma * kNumSteps];
    #pragma unroll
    for (int s = 0; s < kNumSteps; ++ s) {
        #pragma unroll
        for (int r = 0; r < kNumRegPerWgmma; ++ r) {
            int m, k; alayout_64x8(lane, r, m, k);
            a[s * kNumRegPerWgmma + r] = A[m * BLOCK_K + s * MMA::K + k];
        }
    }

    float accum[MMA::kNumAccum] = {};
    auto b_desc = mma90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128, float>(sb, 0, 0);
    const uint32_t b_lo = b_desc.reg32_[0];

    ptx::warpgroup_arrive();
    #pragma unroll
    for (int s = 0; s < kNumSteps; ++ s) {
        b_desc.reg32_[0] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128, float>(
            b_lo, 0, s * MMA::K, 0);
        MMA::wgmma(a + s * kNumRegPerWgmma, b_desc, accum, s == 0 ? 0 : 1);
    }
    ptx::warpgroup_commit_batch();
    ptx::warpgroup_wait<0>();

    // 累加器 CLayout_64xN 的寄存器 -> (m,n) 映射（列主序 offset = n*64 + m）
    const int t0 = lane % 4, t1 = (lane / 4) % 8, t2 = lane / 32;
    #pragma unroll
    for (int r = 0; r < MMA::kNumAccum; ++ r) {
        const int r0 = r % 2, r1 = (r / 2) % 2, r2 = r / 4;
        const int m = t1 + 8 * r1 + 16 * t2;
        const int n = r0 + 2 * t0 + 8 * r2;
        D[m * BLOCK_N + n] = accum[r];
    }
}

// ---------------------------------------------------------------------------
// (4) TMA tf32 转换探针：同一 fp32 张量，用 FLOAT32 vs TFLOAT32 两种 tensor map dtype 载入 smem，
//     直接看 smem 里的字节 —— 判定「fp32→tf32 的舍入」是不是 TMA 做的（而不是 WGMMA）。
//     no-swizzle、no-interleave 的 2D 载入，smem 就是裸行主序。
// ---------------------------------------------------------------------------
__global__ void tma_copy_kernel(const __grid_constant__ CUtensorMap tmap,
                                int box_inner, int box_outer, int elem_bytes, float* out) {
    __shared__ __align__(1024) unsigned char smem[4096];
    __shared__ __align__(8) uint64_t mbar;
    const int tid = threadIdx.x;
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
                     "r"((uint32_t)__cvta_generic_to_shared(&mbar)));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int box_bytes = box_inner * box_outer * elem_bytes;
    if (tid == 0) {
        const uint32_t sp = (uint32_t)__cvta_generic_to_shared(smem);
        const uint32_t mp = (uint32_t)__cvta_generic_to_shared(&mbar);
        const uint64_t dp = reinterpret_cast<uint64_t>(&tmap);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(mp), "r"(box_bytes));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4}], [%2];" ::
            "r"(sp), "l"(dp), "r"(mp), "r"(0), "r"(0) : "memory");
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t mp = (uint32_t)__cvta_generic_to_shared(&mbar);
        asm volatile("{\n\t.reg .pred p;\n\tWAIT: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra WAIT;\n\t}" :: "r"(mp));
    }
    __syncthreads();
    for (int i = tid; i < box_inner * box_outer; i += blockDim.x)
        out[i] = reinterpret_cast<float*>(smem)[i];
}

static CUtensorMap encode_2d(void* ptr, int inner, int outer, CUtensorMapDataType dt) {
    CUtensorMap m;
    cuuint64_t gd[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
    cuuint64_t gs[1] = {(cuuint64_t)inner * 4};
    cuuint32_t sd[2] = {(cuuint32_t)inner, (cuuint32_t)outer};
    cuuint32_t es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(&m, dt, 2, ptr, gd, gs, sd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s); printf("encode err: %s\n", s); exit(1); }
    return m;
}

static void tma_probe(const char* tag, const std::vector<float>& h, int inner, int outer, CUtensorMapDataType dt) {
    float* d; CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    auto map = encode_2d(d, inner, outer, dt);
    float* dout; CUDA_CHECK(cudaMalloc(&dout, h.size() * sizeof(float)));
    tma_copy_kernel<<<1, 32>>>(map, inner, outer, 4, dout);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ho(h.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dout, ho.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::string base = std::string("/tmp/tf32rs_tma_") + tag;
    write_bin((base + "_src.bin").c_str(), h.data(), h.size() * sizeof(float));
    write_bin((base + "_dst.bin").c_str(), ho.data(), ho.size() * sizeof(float));
    printf("TMA  %-10s dtype=%s  -> %s_dst.bin\n", tag, dt == CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 ? "TFLOAT32" : "FLOAT32", base.c_str());
    cudaFree(d); cudaFree(dout);
}

// ---------------------------------------------------------------------------
// host helpers
// ---------------------------------------------------------------------------

static void write_bin(const char* path, const void* p, size_t bytes) {
    FILE* f = fopen(path, "wb"); fwrite(p, 1, bytes, f); fclose(f);
}

template <typename MMA, int BLOCK_N, int BLOCK_K>
static void run_case(const char* tag, const std::vector<float>& hA, const std::vector<float>& hB) {
    constexpr int BLOCK_M = 64;
    float *dA, *dB, *dD;
    CUDA_CHECK(cudaMalloc(&dA, BLOCK_M * BLOCK_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, BLOCK_N * BLOCK_K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(float) * BLOCK_M * BLOCK_N));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    size_t smem = BLOCK_N * BLOCK_K * sizeof(float);
    tf32_rs_gemm<MMA, BLOCK_N, BLOCK_K><<<1, 128, smem>>>(dA, dB, dD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hD(BLOCK_M * BLOCK_N);
    CUDA_CHECK(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::string base = std::string("/tmp/tf32rs_") + tag;
    write_bin((base + "_A.bin").c_str(), hA.data(), hA.size() * sizeof(float));
    write_bin((base + "_B.bin").c_str(), hB.data(), hB.size() * sizeof(float));
    write_bin((base + "_D.bin").c_str(), hD.data(), hD.size() * sizeof(float));
    printf("GEMM %-14s BLOCK_M=64 BLOCK_N=%d BLOCK_K=%d  -> %s_D.bin\n", tag, BLOCK_N, BLOCK_K, base.c_str());

    cudaFree(dA); cudaFree(dB); cudaFree(dD);
}

int main() {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device=%s sm_%d%d\n", prop.name, prop.major, prop.minor);

    // ---- (1) 静态属性 ----
    uint64_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, 16 * sizeof(uint64_t)));
    static_dump<<<1, 1>>>(d_out);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    uint64_t h_out[16] = {};
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    printf("STATIC_BEGIN\n");
    for (int i = 0; i < 12; ++ i) printf("S%02d %llu\n", i, (unsigned long long)h_out[i]);
    printf("STATIC_END\n");
    cudaFree(d_out);

    // ---- (2) A 寄存器布局 ----
    int* d_lay; CUDA_CHECK(cudaMalloc(&d_lay, 128 * 4 * 2 * sizeof(int)));
    alayout_dump<<<1, 128>>>(d_lay);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    int h_lay[128 * 4 * 2] = {};
    CUDA_CHECK(cudaMemcpy(h_lay, d_lay, sizeof(h_lay), cudaMemcpyDeviceToHost));
    printf("LAYOUT_BEGIN\n");
    for (int lane = 0; lane < 128; ++ lane)
        for (int r = 0; r < 4; ++ r)
            printf("L %d %d %d %d\n", lane, r, h_lay[(lane * 4 + r) * 2], h_lay[(lane * 4 + r) * 2 + 1]);
    printf("LAYOUT_END\n");
    cudaFree(d_lay);

    // ---- (3) 真实 WGMMA 数值 ----
    constexpr int K = 32;
    auto make_A_int = [](int m, int k) {
        std::vector<float> a(m * k);
        for (int i = 0; i < m; ++ i) for (int j = 0; j < k; ++ j) a[i * k + j] = (i * 7 + j * 13) % 17 - 8;
        return a;
    };
    auto make_B_int = [](int n, int k) {
        std::vector<float> b(n * k);
        for (int i = 0; i < n; ++ i) for (int j = 0; j < k; ++ j) b[i * k + j] = (i * 5 + j * 11) % 13 - 6;
        return b;
    };
    auto make_A_frac = [](int m, int k) {
        std::vector<float> a(m * k);
        for (int i = 0; i < m; ++ i) for (int j = 0; j < k; ++ j)
            a[i * k + j] = std::sin(i * 0.131f + j * 0.0717f) * 2.5f + std::cos(i * 0.0173f - j * 0.213f);
        return a;
    };
    auto make_B_frac = [](int n, int k) {
        std::vector<float> b(n * k);
        for (int i = 0; i < n; ++ i) for (int j = 0; j < k; ++ j)
            b[i * k + j] = std::cos(i * 0.211f + j * 0.0331f) * 1.7f + std::sin(i * 0.091f + j * 0.417f);
        return b;
    };

    using MMA64 = mma90::TF32MMASelector<64, true>::type;
    using MMA128 = mma90::TF32MMASelector<128, true>::type;

    // N=64
    run_case<MMA64, 64, K>("n64_int",   make_A_int(64, K),  make_B_int(64, K));
    run_case<MMA64, 64, K>("n64_a_frac", make_A_frac(64, K), make_B_int(64, K));
    run_case<MMA64, 64, K>("n64_b_frac", make_A_int(64, K),  make_B_frac(64, K));
    run_case<MMA64, 64, K>("n64_both",   make_A_frac(64, K), make_B_frac(64, K));
    // N=128
    run_case<MMA128, 128, K>("n128_int",  make_A_int(64, K),   make_B_int(128, K));
    run_case<MMA128, 128, K>("n128_both", make_A_frac(64, K),  make_B_frac(128, K));

    // 单元素探针：A[0,0]=1，其余 0；B[n,0]=边界值（trunc 与 rne 不同），其余 0。
    // 于是 D[0,n] == 硬件对该 B 元素的 tf32 量化结果，直接对照 trunc/rne，排除累加干扰。
    {
        std::vector<float> A(64 * K, 0.0f); A[0] = 1.0f;
        std::vector<float> B(64 * K, 0.0f);
        const uint32_t probes[8] = {0x3F801FFF, 0x3F803000, 0x3F805000, 0x3F807000,
                                    0xBF801FFF, 0x3FBFFFFF, 0x40001234, 0x40402345};
        for (int n = 0; n < 8; ++ n) { float v; uint32_t u = probes[n]; std::memcpy(&v, &u, 4); B[n * K + 0] = v; }
        run_case<MMA64, 64, K>("probe", A, B);
    }

    // (4) TMA tf32 转换探针：同一批边界值，两种 tensor map dtype 各载入一次
    {
        std::vector<float> h(64, 0.0f);
        const uint32_t probes[8] = {0x3F801FFF, 0x3F803000, 0x3F805000, 0x3F807000,
                                    0xBF801FFF, 0x3FBFFFFF, 0x40001234, 0x40402345};
        for (int n = 0; n < 8; ++ n) { float v; uint32_t u = probes[n]; std::memcpy(&v, &u, 4); h[n] = v; }
        // 8x8 行主序，inner=8
        tma_probe("f32", h, 8, 8, CU_TENSOR_MAP_DATA_TYPE_FLOAT32);
        tma_probe("tf32", h, 8, 8, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32);
    }

    printf("DONE\n");
    return 0;
}
