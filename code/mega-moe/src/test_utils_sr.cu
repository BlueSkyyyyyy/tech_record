// P33-1 实验（device 侧）：DeepGEMM 测试断言辅助 `tests/utils.py` 的随机舍入参考 oracle。
//
// `tests/utils.py:53-79` 的 `assert_stochastic_bf16_matches_fp32_accumulation` 是 SM100 GEMM
// `epilogue::operators::StochasticRoundToBF16`（`deep_gemm/include/deep_gemm/epilogue/operators.cuh:32-63`）
// 的**测试参考**：它用纯 torch int64 复刻 device 侧那套「quartet 哈希 PRNG + 加随机低位再进位」的
// 随机舍入。本文件把 device 侧算子里的**哈希常数与位选择**逐行原样搬进一个 CUDA kernel，输出
//   (1) 输入 fp32 的 32 位模式；
//   (2) 每个 quartet 的 4 个 16-bit 随机 bit；
//   (3) 随机舍入后的 bf16 输出；
// 供 harness/exp_test_utils.py 与 tests/utils.py 里那条 torch 公式**逐位对拍**——证明测试 oracle 的
// PRNG 位序/常数与真实 device 算子完全一致（第 31 篇已抽出 device 算子本身）。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。
//
// 编译：nvcc -arch=sm_90 -O2 -o test_utils_sr test_utils_sr.cu
// 运行：./test_utils_sr <out_dir> <num_values>   （num_values 必须是 4 的倍数）

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

// ---------------------------------------------------------------------------
// 以下三个函数逐行等价于
//   deep_gemm/include/deep_gemm/epilogue/operators.cuh:35-48 的
//   StochasticRoundToBF16::partial_hash / select_random_bits（+ cast_quartet 的求和）。
// 常数：0x5671d42b / 0x9995e499 / 0xace1b8a5 / 0xe153538d（四个成员的 partial 乘子）、
//       0x7feb352d（最终化乘子）、0x846ca68b（低两个成员）/ 0xd35a2d97（高两个成员）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t partial_hash(const uint32_t v, const uint32_t q) {
    return v * (q == 0 ? 0x5671d42bu : q == 1 ? 0x9995e499u : q == 2 ? 0xace1b8a5u : 0xe153538du);
}

__device__ __forceinline__ uint32_t select_random_bits(uint32_t h, const uint32_t q) {
    h ^= h >> 23;
    h *= 0x7feb352du;
    h ^= h >> 16;
    h *= q < 2 ? 0x846ca68bu : 0xd35a2d97u;
    h ^= h >> 11;
    return (h >> (q % 2 * 16)) & 0xffffu;
}

// 加随机低位再进位、截到高 16 位 = bf16 的随机舍入（与 tests/utils.py:77-78 的
// `rounded = (values >> 16) + (((values & 0xffff) + random_bits) >> 16)` 同构）。
__device__ __forceinline__ uint16_t sr_round(const uint32_t v, const uint32_t rbits) {
    return static_cast<uint16_t>((v >> 16) + (((v & 0xffffu) + rbits) >> 16));
}

// 每个 quartet（4 个连续 fp32）一个线程：算共享哈希 -> 4 个按偏移选的随机 bit -> 4 个 bf16。
// `num_quartets_per_row` 对应把扁平数组看成 (M, N=4*W) 后，N 方向上的 quartet 下标——
// 这正是 `tests/utils.py:66` 的 `quartet_n_idx = arange(shape[-1]//4)`（同一 N 列组在各行复用）。
__global__ void sr_kernel(const uint32_t* __restrict__ in, uint32_t* __restrict__ rbits,
                          uint16_t* __restrict__ out, const int num_quartets,
                          const int num_quartets_per_row) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_quartets) return;
    const uint32_t a = in[q * 4 + 0], b = in[q * 4 + 1], c = in[q * 4 + 2], d = in[q * 4 + 3];
    const int quartet_n_idx = q % num_quartets_per_row;
    const uint32_t h = partial_hash(a, 0) + partial_hash(b, 1) + partial_hash(c, 2) +
                       partial_hash(d, 3) + static_cast<uint32_t>(quartet_n_idx);
    const uint32_t r0 = select_random_bits(h, 0), r1 = select_random_bits(h, 1);
    const uint32_t r2 = select_random_bits(h, 2), r3 = select_random_bits(h, 3);
    rbits[q * 4 + 0] = r0; rbits[q * 4 + 1] = r1; rbits[q * 4 + 2] = r2; rbits[q * 4 + 3] = r3;
    out[q * 4 + 0] = sr_round(a, r0); out[q * 4 + 1] = sr_round(b, r1);
    out[q * 4 + 2] = sr_round(c, r2); out[q * 4 + 3] = sr_round(d, r3);
}

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <out_dir> <num_values> <num_quartets_per_row>\n", argv[0]); return 1; }
    const char* out_dir = argv[1];
    const int num_values = std::atoi(argv[2]);
    const int num_quartets_per_row = std::atoi(argv[3]);
    if (num_values <= 0 || num_values % 4 != 0) { printf("num_values must be positive multiple of 4\n"); return 1; }
    if (num_quartets_per_row <= 0 || (num_values / 4) % num_quartets_per_row != 0) {
        printf("num_values/4 must be a multiple of num_quartets_per_row\n"); return 1; }
    const int num_quartets = num_values / 4;

    // 确定性输入：复用 DeepGEMM 那套 32-bit 线性同余（乘 1664525 加 1013904223），
    // 但把结果规约到 [-1,1) 附近的有限 fp32（避免 inf/NaN，随机舍入有意义的范围）。
    std::vector<uint32_t> h_in(num_values);
    uint64_t state = 20261013ull;
    for (int i = 0; i < num_values; i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const uint32_t r = static_cast<uint32_t>(state >> 33);
        // 取 23 位尾数 + 指数域，构造量级 2^-8..2^0 的有限正负数
        const float x = (static_cast<float>(r & 0x7fffffu) / static_cast<float>(0x800000) - 0.5f);
        const int e = 120 + static_cast<int>((r >> 23) % 8);  // 2^-7 .. 2^0
        float v = ldexpf(x, e - 127);
        if (v == 0.0f) v = ldexpf(1.0f, e - 127);  // 避免恰好 0，保证有尾数位
        uint32_t u; __builtin_memcpy(&u, &v, 4);
        h_in[i] = u;
    }

    uint32_t *d_in, *d_rb; uint16_t* d_out;
    CK(cudaMalloc(&d_in, num_values * sizeof(uint32_t)));
    CK(cudaMalloc(&d_rb, num_values * sizeof(uint32_t)));
    CK(cudaMalloc(&d_out, num_values * sizeof(uint16_t)));
    CK(cudaMemcpy(d_in, h_in.data(), num_values * sizeof(uint32_t), cudaMemcpyHostToDevice));

    const int threads = 256, blocks = (num_quartets + threads - 1) / threads;
    sr_kernel<<<blocks, threads>>>(d_in, d_rb, d_out, num_quartets, num_quartets_per_row);
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> h_rb(num_values);
    std::vector<uint16_t> h_out(num_values);
    CK(cudaMemcpy(h_rb.data(), d_rb, num_values * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_out.data(), d_out, num_values * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CK(cudaFree(d_in)); CK(cudaFree(d_rb)); CK(cudaFree(d_out));

    char path[1024];
    FILE* f;
    snprintf(path, sizeof(path), "%s/sr_in.bin", out_dir);   f = fopen(path, "wb");
    fwrite(h_in.data(), sizeof(uint32_t), num_values, f); fclose(f);
    snprintf(path, sizeof(path), "%s/sr_rb.bin", out_dir);   f = fopen(path, "wb");
    fwrite(h_rb.data(), sizeof(uint32_t), num_values, f); fclose(f);
    snprintf(path, sizeof(path), "%s/sr_out.bin", out_dir);  f = fopen(path, "wb");
    fwrite(h_out.data(), sizeof(uint16_t), num_values, f); fclose(f);

    printf("sr_kernel: wrote %d values (%d quartets) to %s\n", num_values, num_quartets, out_dir);
    return 0;
}
