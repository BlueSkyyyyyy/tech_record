// P60-1 负样本控制：故意制造 memcheck / synccheck 能命中的错误，验证
//   tests/test_sanitizer.py 的驱动方式与那组 flag 真的在“发现问题”。
// 对应 harness/exp_sanitizer.py；参考 ~/github/DeepGEMM/tests/test_sanitizer.py。
//
// 用法：sanitizer_negctl <mode>
//   clean      干净 kernel（既不越界也无竞争）
//   oob        越界写（第 n 个元素）           -> memcheck
//   misalign   共享内存非对齐访问               -> memcheck
//   sync       发散分支里的 __syncthreads      -> synccheck
//   race       共享内存无同步竞争               -> racecheck（注：test_sanitizer 并不跑它）
//   mask_big   写进“大段里紧邻下一块”的位置      -> memcheck 报 0（模拟缓存分配器）
//   mask_exact 精确分配后越界写                  -> memcheck 命中（模拟关闭缓存分配器）
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)

__global__ void clean_kernel(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = d[i] + 1.0f;
}

__global__ void oob_write_kernel(float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 故意越界：i == n 的线程写 d[n]
    if (i == n) d[i] = 42.0f;
    else if (i < n) d[i] = d[i] + 1.0f;
}

__global__ void race_kernel(int* out) {
    __shared__ int s;
    if (threadIdx.x == 0) s = 1;
    if (threadIdx.x == 1) s = 2;  // 故意：无同步的 WAW 竞争
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// 写第 n 个元素（越界）；用于模拟“越界恰好落进缓存分配器的同一大段”
__global__ void oob_n_kernel(float* p, long n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) p[n] = 42.0f;
}

__global__ void sync_kernel(int* out) {
    // 故意：只在半数线程里执行 barrier（发散 barrier），非法。
    // 两侧都有副作用，ptxas 不能把 barrier 分支 if-convert/提前 EXIT。
    int t = threadIdx.x;
    if (t < 16) {
        asm volatile("bar.sync 1, 32;");
        out[blockIdx.x * 32 + t] = 1;
    } else {
        out[blockIdx.x * 32 + t] = 2;
    }
}

__global__ void misalign_kernel(float* out) {
    __shared__ char buf[128];
    // 4 字节偏移后按 8 字节 double 访问 -> 非 8 对齐
    double* dp = (double*)(buf + 4);
    *dp = 2.0;
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = (float)*dp;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "clean";
    const int n = 256;
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));

    if (strcmp(mode, "clean") == 0) {
        clean_kernel<<<(n + 63) / 64, 64>>>(d, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("clean: done\n");
    } else if (strcmp(mode, "oob") == 0) {
        oob_write_kernel<<<(n + 2 + 63) / 64, 64>>>(d, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("oob: done\n");
    } else if (strcmp(mode, "race") == 0) {
        int* out = nullptr;
        CUDA_CHECK(cudaMalloc(&out, 4 * sizeof(int)));
        race_kernel<<<4, 256>>>(out);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("race: done\n");
        CUDA_CHECK(cudaFree(out));
    } else if (strcmp(mode, "sync") == 0) {
        int* out = nullptr;
        CUDA_CHECK(cudaMalloc(&out, 4 * 32 * sizeof(int)));
        sync_kernel<<<4, 32>>>(out);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("sync: done\n");
        CUDA_CHECK(cudaFree(out));
    } else if (strcmp(mode, "mask_big") == 0) {
        // 一段 2n 的大分配：x = [0,n)，y 紧邻其后；写 x[n] 落在同段内 → memcheck 无感
        float* seg = nullptr;
        CUDA_CHECK(cudaMalloc(&seg, 2L * n * sizeof(float)));
        CUDA_CHECK(cudaMemset(seg, 0, 2L * n * sizeof(float)));
        oob_n_kernel<<<1, 32>>>(seg, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("mask_big: done\n");
        CUDA_CHECK(cudaFree(seg));
    } else if (strcmp(mode, "mask_exact") == 0) {
        // 精确 n 个 float 的分配：写 x[n] 越界 → memcheck 命中
        float* x = nullptr;
        CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(x, 0, n * sizeof(float)));
        oob_n_kernel<<<1, 32>>>(x, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("mask_exact: done\n");
        CUDA_CHECK(cudaFree(x));
    } else if (strcmp(mode, "misalign") == 0) {
        float* out = nullptr;
        CUDA_CHECK(cudaMalloc(&out, 4 * sizeof(float)));
        misalign_kernel<<<4, 32>>>(out);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("misalign: done\n");
        CUDA_CHECK(cudaFree(out));
    } else {
        printf("unknown mode: %s\n", mode);
        return 1;
    }
    CUDA_CHECK(cudaFree(d));
    return 0;
}
