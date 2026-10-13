// P57-1 实验：Mega mHC 的验收测试用的 SF 页布局行索引公式，与 kernel 真实设备代码对拍。
//
// 来自 tests/test_mega_mhc.py 的 `extra_sf_rows`（arch 无关）：
//   row = token // block_m * align(block_m,128)
//       + token % block_m // 128 * 128
//       + token % block_m % 32 * 4
//       + token % block_m % 128 // 32
//
// 来自 deep_gemm/include/deep_gemm/epilogue/sm100_mega_mhc.cuh:116-122 的
// `get_shared_sf_row_idx`（**直接 include 真实头文件调用真实设备代码**）：
//   #define idx = token % SF_BLOCK_M
//   row = token / SF_BLOCK_M * ((SF_BLOCK_M+127)&~127) + (idx & ~127u)
//       + (idx & 31u) * 4 + ((idx >> 5) & 3u)
//
// 对若干个 SF_BLOCK_M（含测试默认 224 与 64/128/240/512）与 token 数，
// 逐 token 比较两者；并验证映射在每个 (block, hidden-word) 内是单射。
//
// 编译运行（H100/sm90，nvcc）：
//   nvcc -std=c++17 -arch=sm_90a \
//     -I$DEEPGEMM/deep_gemm/include -I$DEEPGEMM/third-party/cutlass/include \
//     src/mhc_sf_row_idx.cu -o /tmp/mhc_sf_row_idx
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

// 该真实头文件里 `normalize_bf16_pair` 用了 SM100 专属的 `__fmul2_rn`（packed fp32x2）。
// 我们只调用架构无关的 `get_shared_sf_row_idx`，故在 sm90 设备编译期给 `__fmul2_rn`
// 提供一个语义等价的本地回退（两次标量 `__fmul_rn`），不改动被调用的函数。
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
__device__ __forceinline__ float2 __fmul2_rn(const float2& a, const float2& b) {
    return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));
}
#endif

#include <deep_gemm/epilogue/sm100_mega_mhc.cuh>

using deep_gemm::epilogue::mega_mhc::get_shared_sf_row_idx;

static __device__ __forceinline__ uint32_t test_extra_sf_rows(uint32_t token, uint32_t block_m) {
    const uint32_t aligned = (block_m + 127u) & ~127u;
    const uint32_t idx = token % block_m;
    return token / block_m * aligned + (idx / 128u) * 128u + (idx % 32u) * 4u + (idx % 128u) / 32u;
}

template <uint32_t SF_BLOCK_M>
static __global__ void compute_rows(const uint32_t num_tokens, uint32_t* kernel_rows, uint32_t* test_rows) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    kernel_rows[t] = get_shared_sf_row_idx<SF_BLOCK_M>(t);
    test_rows[t] = test_extra_sf_rows(t, SF_BLOCK_M);
}

static int PASS = 0, FAIL = 0;
static void check(const char* name, bool cond, const char* extra = "") {
    (cond ? PASS : FAIL)++;
    printf("  [%s] %s %s\n", cond ? "PASS" : "FAIL", name, extra);
}

template <uint32_t SF_BLOCK_M>
static void run_case(uint32_t num_tokens) {
    uint32_t *dk = nullptr, *dt = nullptr, *hk = new uint32_t[num_tokens], *ht = new uint32_t[num_tokens];
    cudaMalloc(&dk, num_tokens * sizeof(uint32_t));
    cudaMalloc(&dt, num_tokens * sizeof(uint32_t));
    compute_rows<SF_BLOCK_M><<<(num_tokens + 255) / 256, 256>>>(num_tokens, dk, dt);
    cudaDeviceSynchronize();
    cudaMemcpy(hk, dk, num_tokens * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ht, dt, num_tokens * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t mismatch = 0;
    for (uint32_t i = 0; i < num_tokens; ++i) mismatch += (hk[i] != ht[i]);

    // 每个 128-行页内单射（同一页里不同 token 必落到不同行）。
    const uint32_t aligned = (SF_BLOCK_M + 127u) & ~127u;
    const uint32_t words_per_row = 1;  // 只看行号本身
    std::vector<int> seen;
    uint32_t inj_bad = 0;
    for (uint32_t base = 0; base < num_tokens; base += SF_BLOCK_M) {
        uint32_t cnt = std::min<uint32_t>(SF_BLOCK_M, num_tokens - base);
        std::vector<int> occ(aligned, 0);
        for (uint32_t i = 0; i < cnt; ++i) {
            uint32_t r = hk[base + i] - base / SF_BLOCK_M * aligned;
            if (r >= aligned) { inj_bad++; continue; }
            if (occ[r]++) inj_bad++;
        }
    }
    (void)words_per_row;
    char extra[160];
    snprintf(extra, sizeof(extra), "SF_BLOCK_M=%u T=%u mismatch=%u inj_bad=%u",
             SF_BLOCK_M, num_tokens, mismatch, inj_bad);
    check("kernel==test formula", mismatch == 0, extra);
    check("injective in page", inj_bad == 0, extra);
    cudaFree(dk); cudaFree(dt); delete[] hk; delete[] ht;
}

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    printf("device=%s cc=%d.%d\n", prop.name, prop.major, prop.minor);
    printf("\n== kernel get_shared_sf_row_idx<SF_BLOCK_M> vs test extra_sf_rows ==\n");
    run_case<64>(1); run_case<64>(200); run_case<64>(1025);
    run_case<128>(1); run_case<128>(200); run_case<128>(1025);
    run_case<224>(1); run_case<224>(65); run_case<224>(200); run_case<224>(1025); run_case<224>(4096); run_case<224>(32768);
    run_case<240>(200); run_case<240>(4096);
    run_case<512>(1025); run_case<512>(32768);
    printf("\nTOTAL: PASS=%d FAIL=%d\n", PASS, FAIL);
    return FAIL ? 1 : 0;
}
