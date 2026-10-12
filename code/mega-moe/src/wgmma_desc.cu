// P39-1 实验：DeepGEMM SM90 WGMMA 层（mma/sm90.cuh + ptx/wgmma.cuh）的共享内存描述符与选择器。
//
// 为什么 H100(sm90a) 可跑：`deep_gemm/include/deep_gemm/mma/sm90.cuh` 的
//   make_smem_desc / make_gmma_desc / advance_gmma_desc_lo / *_Selector
// 与 `deep_gemm/include/deep_gemm/ptx/wgmma.cuh` 的 fence/commit/wait 都是 SM90 设备代码，
// 本机 H100 直接编译运行真实 WGMMA 指令。
//
// 本文件：
//   (1) desc_dump     —— 原样调用真实描述符构造，dump 64-bit 描述符，Python 独立复刻位域核对。
//   (2) wgmma_gemm_*  —— 用手工按 TMA 128B swizzle 铺好的 smem，调用真实选择器 + 真实 WGMMA，
//                        结果按 CLayout_64xN 的（推导出的）寄存器->(m,n) 映射落盘，Python 对参考 GEMM。
//   (3) 选择器静态属性与 nvcc 反汇编冒烟。
//
// 编译：nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr ...
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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

// ---------------------------------------------------------------------------
// (1) 描述符 dump：单线程，把若干真实描述符写进全局
// ---------------------------------------------------------------------------
__global__ void desc_dump(uint64_t* out) {
    extern __shared__ __align__(1024) uint8_t smem[];
    auto* p = reinterpret_cast<nv_bfloat16*>(smem);

    // make_smem_desc：显式 layout_type
    out[0] = mma90::make_smem_desc(p, 0);                       // INTERLEAVE, 默认 LBO=0 SBO=1024
    out[1] = mma90::make_smem_desc(p, 1);                       // B128,      默认 LBO=0 SBO=1024
    out[2] = mma90::make_smem_desc(p, 1, 0x1230, 0x4560);       // B128, 显式 LBO/SBO（16 的倍数）
    out[3] = mma90::make_smem_desc(p, 3, 16, 112);              // B32

    // make_gmma_desc：K-major，bf16，BLOCK_K=64（=128B swizzle）
    out[4] = mma90::make_gmma_desc<cute::UMMA::Major::K, 64, 64, 128, nv_bfloat16>(p, 0, 0);
    out[5] = mma90::make_gmma_desc<cute::UMMA::Major::K, 64, 64, 128, nv_bfloat16>(p, 16, 0);
    // MN-major，bf16，BLOCK_K=64，swizzle 128：BLOCK_MN_ATOM = 128/2 = 64
    out[6] = mma90::make_gmma_desc<cute::UMMA::Major::MN, 64, 64, 128, nv_bfloat16>(p, 0, 0);
    // MN-major，非 swizzle 的 interleave 分支（kSwizzleMode==16 会 swap LBO/SBO）
    out[7] = mma90::make_gmma_desc<cute::UMMA::Major::MN, 64, 64, 16, nv_bfloat16>(p, 0, 0);
    // FP8，K-major，BLOCK_K=128（=128B）
    out[8] = mma90::make_gmma_desc<cute::UMMA::Major::K, 64, 128, 128, __nv_fp8_e4m3>(
        reinterpret_cast<__nv_fp8_e4m3*>(smem), 0, 0);

    // advance_gmma_desc_lo：返回的是低 32 位 uint32
    const uint32_t k_base = static_cast<uint32_t>(mma90::make_gmma_desc<cute::UMMA::Major::K, 64, 64, 128, nv_bfloat16>(p, 0, 0).reg32_[0]);
    out[9]  = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, 64, 64, 128, nv_bfloat16>(k_base, 0, 0, 0);
    out[10] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, 64, 64, 128, nv_bfloat16>(k_base, 0, 16, 0);
    out[11] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, 64, 64, 128, nv_bfloat16>(k_base, 16, 0, 0);
    out[12] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, 64, 64, 128, nv_bfloat16>(k_base, 0, 0, 64 * 64);
    const uint32_t mn_base = static_cast<uint32_t>(mma90::make_gmma_desc<cute::UMMA::Major::MN, 64, 64, 128, nv_bfloat16>(p, 0, 0).reg32_[0]);
    out[13] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::MN, 64, 64, 128, nv_bfloat16>(mn_base, 0, 4, 0);
    // k_idx=16（bf16 单步 K=16）→ +32B = +2 个 16B 单位；k_idx=4（8B）被 >>4 截断为 0
    out[28] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, 64, 64, 128, nv_bfloat16>(k_base, 0, 4, 0);

    // 选择器静态属性
    out[14] = mma90::BF16MMASelector<128>::type::M;
    out[15] = mma90::BF16MMASelector<128>::type::N;
    out[16] = mma90::BF16MMASelector<128>::type::K;
    out[17] = mma90::BF16MMASelector<128>::type::kNumAccum;
    out[18] = mma90::FP8MMASelector<128>::type::M;
    out[19] = mma90::FP8MMASelector<128>::type::K;
    out[20] = mma90::FP8MMASelector<128>::type::kNumAccum;
    out[21] = mma90::TF32MMASelector<128>::type::M;
    out[22] = mma90::TF32MMASelector<128>::type::K;
    out[23] = mma90::TF32MMASelector<128>::type::kNumAccum;
    out[24] = mma90::get_gmma_desc_stride_k<cute::UMMA::Major::K, 64, 128, nv_bfloat16>();
    out[25] = mma90::get_gmma_desc_stride_k<cute::UMMA::Major::MN, 64, 128, nv_bfloat16>();
    out[26] = mma90::get_inner_block_atom_size<64, 32, nv_bfloat16>();
    out[27] = mma90::get_inner_block_atom_size<64, 0, nv_bfloat16>();
}

// ---------------------------------------------------------------------------
// (2) 真实 WGMMA GEMM：A[BLOCK_M,K] @ B[BLOCK_N,K]^T
//   K == BLOCK_K，每行 128 字节；smem 用 TMA 128B swizzle 铺。
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void fill_swizzle_128(T* smem, const T* glob, int rows, int k) {
    constexpr int EPB = 16 / sizeof(T);           // 每个 16B chunk 的元素数
    for (int i = threadIdx.x; i < rows * k; i += blockDim.x) {
        int row = i / k, col = i % k;
        int chunk = col / EPB, within = col % EPB;
        int chunk2 = chunk ^ (row & 7);           // SWIZZLE_128B：16B chunk 下标 XOR (row % 8)
        smem[row * k + chunk2 * EPB + within] = glob[row * k + col];
    }
}

template <typename MMA, typename T, int BLOCK_N, int BLOCK_K, bool kSplitCommit = false>
__global__ void wgmma_gemm(const T* A, const T* B, float* D) {
    constexpr int BLOCK_M = 64;                   // 单个 warpgroup 的 M
    extern __shared__ __align__(1024) uint8_t smem[];
    T* sa = reinterpret_cast<T*>(smem);
    T* sb = reinterpret_cast<T*>(smem + BLOCK_M * BLOCK_K * sizeof(T));

    fill_swizzle_128<T>(sa, A, BLOCK_M, BLOCK_K);
    fill_swizzle_128<T>(sb, B, BLOCK_N, BLOCK_K);
    __syncthreads();

    float accum[MMA::kNumAccum] = {};
    auto a_desc = mma90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, 128, T>(sa, 0, 0);
    auto b_desc = mma90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128, T>(sb, 0, 0);
    const uint32_t a_lo = a_desc.reg32_[0], b_lo = b_desc.reg32_[0];
    constexpr int kNumSteps = BLOCK_K / MMA::K;
    constexpr int kHalf = kNumSteps / 2;

    ptx::warpgroup_arrive();
    #pragma unroll
    for (int k = 0; k < kNumSteps; ++ k) {
        a_desc.reg32_[0] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, 128, T>(
            a_lo, 0, (k * MMA::K) % BLOCK_K, 0);
        b_desc.reg32_[0] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128, T>(
            b_lo, 0, (k * MMA::K) % BLOCK_K, 0);
        MMA::wgmma(a_desc, b_desc, accum, k == 0 ? 0 : 1);
        if constexpr (kSplitCommit) {
            if (k == kHalf - 1) {
                ptx::warpgroup_commit_batch();
                ptx::warpgroup_wait<1>();       // 允许 1 组未完成，再发下一组
                ptx::warpgroup_arrive();
            }
        }
    }
    ptx::warpgroup_commit_batch();
    ptx::warpgroup_wait<0>();

    // CLayout_64xN 的寄存器 -> (m,n) 映射（列主序：offset = n*64 + m）
    const int lane = threadIdx.x;
    const int t0 = lane % 4, t1 = (lane / 4) % 8, t2 = lane / 32;
    #pragma unroll
    for (int r = 0; r < MMA::kNumAccum; ++ r) {
        const int r0 = r % 2, r1 = (r / 2) % 2, r2 = r / 4;
        const int m = t1 + 8 * r1 + 16 * t2;
        const int n = r0 + 2 * t0 + 8 * r2;
        D[m * BLOCK_N + n] = accum[r];
    }
}

// A 为 MN-major（全局 A[k][m]），B 为 K-major；D[m][n]
template <typename MMA, typename T, int BLOCK_N, int BLOCK_K>
__global__ void wgmma_gemm_mn_a(const T* A, const T* B, float* D) {
    constexpr int BLOCK_M = 64;
    constexpr int EPB = 16 / sizeof(T);
    extern __shared__ __align__(1024) uint8_t smem[];
    T* sa = reinterpret_cast<T*>(smem);
    T* sb = reinterpret_cast<T*>(smem + BLOCK_M * BLOCK_K * sizeof(T));

    // MN-major：物理行 = M 维（64 元素 = 128B），K 为外层；SWIZZLE_128B 按 16B chunk XOR(k%8)
    for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
        int k = i / BLOCK_M, m = i % BLOCK_M;
        int chunk = (m / EPB) ^ (k & 7), within = m % EPB;
        sa[k * BLOCK_M + chunk * EPB + within] = A[k * BLOCK_M + m];
    }
    fill_swizzle_128<T>(sb, B, BLOCK_N, BLOCK_K);
    __syncthreads();

    float accum[MMA::kNumAccum] = {};
    auto a_desc = mma90::make_gmma_desc<cute::UMMA::Major::MN, BLOCK_M, BLOCK_K, 128, T>(sa, 0, 0);
    auto b_desc = mma90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128, T>(sb, 0, 0);
    const uint32_t a_lo = a_desc.reg32_[0], b_lo = b_desc.reg32_[0];

    ptx::warpgroup_arrive();
    #pragma unroll
    for (int k = 0; k < BLOCK_K / MMA::K; ++ k) {
        a_desc.reg32_[0] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::MN, BLOCK_M, BLOCK_K, 128, T>(
            a_lo, 0, k * MMA::K, 0);
        b_desc.reg32_[0] = mma90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128, T>(
            b_lo, 0, (k * MMA::K) % BLOCK_K, 0);
        MMA::wgmma(a_desc, b_desc, accum, k == 0 ? 0 : 1);
    }
    ptx::warpgroup_commit_batch();
    ptx::warpgroup_wait<0>();

    const int lane = threadIdx.x;
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
// host helpers
// ---------------------------------------------------------------------------
template <typename T>
static void to_device(const std::vector<float>& h, T* d) {
    std::vector<T> t(h.size());
    for (size_t i = 0; i < h.size(); ++ i) t[i] = static_cast<T>(h[i]);
    CUDA_CHECK(cudaMemcpy(d, t.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

static void write_bin(const char* path, const void* p, size_t bytes) {
    FILE* f = fopen(path, "wb");
    fwrite(p, 1, bytes, f);
    fclose(f);
}

template <typename MMA, typename T, int BLOCK_N, int BLOCK_K, bool kSplitCommit = false>
static void run_gemm(const char* tag, const std::vector<float>& hA, const std::vector<float>& hB,
                     const char* fa, const char* fb, const char* fd) {
    constexpr int BLOCK_M = 64;
    constexpr int K = BLOCK_K;
    T *dA, *dB; float* dD;
    CUDA_CHECK(cudaMalloc(&dA, BLOCK_M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dB, BLOCK_N * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(float) * BLOCK_M * BLOCK_N));
    to_device<T>(hA, dA);
    to_device<T>(hB, dB);

    size_t smem = (BLOCK_M * K + BLOCK_N * K) * sizeof(T);
    wgmma_gemm<MMA, T, BLOCK_N, K, kSplitCommit><<<1, 128, smem>>>(dA, dB, dD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hD(BLOCK_M * BLOCK_N);
    CUDA_CHECK(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(float), cudaMemcpyDeviceToHost));
    write_bin(fd, hD.data(), hD.size() * sizeof(float));
    write_bin(fa, hA.data(), hA.size() * sizeof(float));
    write_bin(fb, hB.data(), hB.size() * sizeof(float));
    printf("GEMM %s: BLOCK_M=%d BLOCK_N=%d BLOCK_K=%d done (D written to %s)\n", tag, BLOCK_M, BLOCK_N, K, fd);

    cudaFree(dA); cudaFree(dB); cudaFree(dD);
}

template <typename MMA, typename T, int BLOCK_N, int BLOCK_K>
static void run_gemm_mn(const char* tag, const std::vector<float>& hA_t, const std::vector<float>& hB,
                        const char* fa, const char* fb, const char* fd) {
    constexpr int BLOCK_M = 64;
    constexpr int K = BLOCK_K;
    // hA_t 是 [K, M]（MN-major），转成 kernel 期望的 [K,M] 行主序（k*BLOCK_M+m）
    T *dA, *dB; float* dD;
    CUDA_CHECK(cudaMalloc(&dA, BLOCK_M * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dB, BLOCK_N * K * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(float) * BLOCK_M * BLOCK_N));
    to_device<T>(hA_t, dA);
    to_device<T>(hB, dB);

    size_t smem = (BLOCK_M * K + BLOCK_N * K) * sizeof(T);
    wgmma_gemm_mn_a<MMA, T, BLOCK_N, K><<<1, 128, smem>>>(dA, dB, dD);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hD(BLOCK_M * BLOCK_N);
    CUDA_CHECK(cudaMemcpy(hD.data(), dD, hD.size() * sizeof(float), cudaMemcpyDeviceToHost));
    write_bin(fd, hD.data(), hD.size() * sizeof(float));
    write_bin(fa, hA_t.data(), hA_t.size() * sizeof(float));
    // 参考脚本需要 A[m][k]：这里额外把 A[k][m] 转置写一份
    std::vector<float> hA_mk(BLOCK_M * K);
    for (int k = 0; k < K; ++ k) for (int m = 0; m < BLOCK_M; ++ m) hA_mk[m * K + k] = hA_t[k * BLOCK_M + m];
    write_bin(fa, hA_mk.data(), hA_mk.size() * sizeof(float));
    write_bin(fb, hB.data(), hB.size() * sizeof(float));
    printf("GEMM %s: BLOCK_M=%d BLOCK_N=%d BLOCK_K=%d done (D written to %s)\n", tag, BLOCK_M, BLOCK_N, K, fd);

    cudaFree(dA); cudaFree(dB); cudaFree(dD);
}

int main() {
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device=%s sm_%d%d\n", prop.name, prop.major, prop.minor);

    // ---- (1) 描述符 dump ----
    uint64_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, 64 * sizeof(uint64_t)));
    desc_dump<<<1, 1, 4096>>>(d_out);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    uint64_t h_out[64] = {};
    CUDA_CHECK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));
    printf("DESC_BEGIN\n");
    for (int i = 0; i < 29; ++ i) printf("D%02d 0x%016llx\n", i, (unsigned long long)h_out[i]);
    printf("DESC_END\n");
    cudaFree(d_out);

    // ---- (2) 真实 WGMMA 数值 ----
    // A[m][k], B[n][k] 小整数，bf16/fp8 可精确表示
    auto make_A = [](int m, int k) {
        std::vector<float> a(m * k);
        for (int i = 0; i < m; ++ i) for (int j = 0; j < k; ++ j) a[i * k + j] = (i * 7 + j * 13) % 17 - 8;
        return a;
    };
    auto make_B = [](int n, int k) {
        std::vector<float> b(n * k);
        for (int i = 0; i < n; ++ i) for (int j = 0; j < k; ++ j) b[i * k + j] = (i * 5 + j * 11) % 13 - 6;
        return b;
    };

    // BF16, BLOCK_N=128, BLOCK_K=64
    {
        using MMA = mma90::BF16MMASelector<128, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
        run_gemm<MMA, nv_bfloat16, 128, 64>("bf16_n128_k64", make_A(64, 64), make_B(128, 64),
            "/tmp/wgmma_A_bf16.bin", "/tmp/wgmma_B_bf16.bin", "/tmp/wgmma_D_bf16.bin");
    }
    // BF16, BLOCK_N=64, BLOCK_K=64（更小 N）
    {
        using MMA = mma90::BF16MMASelector<64, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
        run_gemm<MMA, nv_bfloat16, 64, 64>("bf16_n64_k64", make_A(64, 64), make_B(64, 64),
            "/tmp/wgmma_A_bf16b.bin", "/tmp/wgmma_B_bf16b.bin", "/tmp/wgmma_D_bf16b.bin");
    }
    // FP8, BLOCK_N=128, BLOCK_K=128
    {
        using MMA = mma90::FP8MMASelector<128>::type;
        run_gemm<MMA, __nv_fp8_e4m3, 128, 128>("fp8_n128_k128", make_A(64, 128), make_B(128, 128),
            "/tmp/wgmma_A_fp8.bin", "/tmp/wgmma_B_fp8.bin", "/tmp/wgmma_D_fp8.bin");
    }
    // BF16, 分两批 commit + wait<1>（异步组语义）
    {
        using MMA = mma90::BF16MMASelector<128, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
        run_gemm<MMA, nv_bfloat16, 128, 64, /*kSplitCommit=*/true>("bf16_split", make_A(64, 64), make_B(128, 64),
            "/tmp/wgmma_A_bf16s.bin", "/tmp/wgmma_B_bf16s.bin", "/tmp/wgmma_D_bf16s.bin");
    }
    // BF16, A 为 MN-major（kMajorA=MN）
    {
        using MMA = mma90::BF16MMASelector<128, cute::UMMA::Major::MN, cute::UMMA::Major::K>::type;
        auto A_mk = make_A(64, 64);                  // [M,K]，K=64（bf16 swizzle 128 要求 BLOCK_K=64）
        std::vector<float> A_km(64 * 64);
        for (int m = 0; m < 64; ++ m) for (int k = 0; k < 64; ++ k) A_km[k * 64 + m] = A_mk[m * 64 + k];
        run_gemm_mn<MMA, nv_bfloat16, 128, 64>("bf16_mn_a", A_km, make_B(128, 64),
            "/tmp/wgmma_A_mn.bin", "/tmp/wgmma_B_mn.bin", "/tmp/wgmma_D_mn.bin");
    }
    printf("ALL_DONE\n");
    return 0;
}
