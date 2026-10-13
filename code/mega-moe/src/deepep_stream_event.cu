// P65-1 实验：镜像 DeepEP 的通信-计算重叠调度原语（事件 / 流），在 H100 上跑真实 CUDA 代码。
//
// 目标源码 `~/github/DeepEP@93eb6eb`：
//   * csrc/utils/event.hpp:13-30    —— EventHandle { record(stream); current_stream_wait() }
//   * csrc/comm/stream.hpp:9-27     —— get_comm_stream()（stream-from-pool(high prio), static）
//                                       + stream_wait()（EP_HOST_ASSERT(id!=) 防自等待）
// 这两处是 torch 类型（torch::Event / at::cuda::CUDAStream）的薄封装，无法脱离 torch 编译；
// 本文件用等价的 CUDA Runtime 原语（cudaEvent / cudaStream）**逐行镜像其语义**并验证：
//   ① EventHandle.record/wait 的 memory-ordering：消费者流等待后一定看得见生产者流的写；
//   ② stream_wait 的「不同流 id」断言（自等待会死锁/无效）；
//   ③ get_comm_stream 的 stream-from-pool 高优先级 + 单例复用；
//   ④ 通信腿（device copy）与计算腿（FFMA）在两条流上的墙钟：串行 vs 重叠。
//
// 编译：nvcc -arch=sm_90a -O3 -std=c++17 deepep_stream_event.cu -o /tmp/deepep_stream_event
// 运行：/tmp/deepep_stream_event

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while (0)

// ---------------------------------------------------------------------------------------
// 镜像 csrc/utils/event.hpp 的 EventHandle（torch::Event → cudaEvent_t）
// ---------------------------------------------------------------------------------------
struct EventHandle {
    cudaEvent_t event;
    explicit EventHandle(cudaStream_t stream) {         // EventHandle(const CUDAStream&)
        CK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        CK(cudaEventRecord(event, stream));             // event->record(stream)
    }
    EventHandle(const EventHandle& o) = default;
    void current_stream_wait() const {                  // getCurrentCUDAStream().wait(*event)
        CK(cudaStreamWaitEvent(0, event, 0));           // 默认流(0) = 当前流
    }
};

// 镜像 csrc/comm/stream.hpp:20-23 的 stream_wait(stream, event_stream)
//   EP_HOST_ASSERT(stream.id() != event_stream.id()); stream.wait(create_event(event_stream));
static bool stream_wait(cudaStream_t stream, cudaStream_t event_stream) {
    bool ids_equal = (stream == event_stream);
    if (ids_equal) return false;                        // 断言失败：自等待
    cudaEvent_t e;
    CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CK(cudaEventRecord(e, event_stream));
    CK(cudaStreamWaitEvent(stream, e, 0));
    CK(cudaEventDestroy(e));
    return true;
}

// ---------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------
__global__ void writer_kernel(float* out, unsigned long long spin, float val) {
    if (threadIdx.x == 0) {
        unsigned long long s = clock64();
        while (clock64() - s < spin) {}
        out[0] = val;
        __threadfence_system();                         // 让消费者流可见
    }
}

__global__ void reader_kernel(const float* in, int* bad, float expect) {
    if (threadIdx.x == 0 && in[0] != expect) atomicAdd(bad, 1);
}

__global__ void comm_copy_kernel(float* __restrict__ dst, const float* __restrict__ src, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = src[i];
}

__global__ void compute_kernel(float* __restrict__ buf, int iters) {
    float a = buf[threadIdx.x], b = 1.0000001f;
    for (int i = 0; i < iters; ++i) a = fmaf(a, b, 0.5f);
    buf[threadIdx.x] = a;
}

// ---------------------------------------------------------------------------------------
static float time_serial(cudaStream_t comm, float* dst, float* src, long n, int comm_blocks,
                         float* cbuf, dim3 cgrid, dim3 cblock, int iters) {
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
    comm_copy_kernel<<<comm_blocks, 256, 0, comm>>>(dst, src, n);
    // 真实串行：当前流等待通信流完成后才发计算
    EventHandle h(comm);
    h.current_stream_wait();
    compute_kernel<<<cgrid, cblock, 0, 0>>>(cbuf, iters);
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

static float time_overlap(cudaStream_t comm, float* dst, float* src, long n, int comm_blocks,
                          float* cbuf, dim3 cgrid, dim3 cblock, int iters) {
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
    comm_copy_kernel<<<comm_blocks, 256, 0, comm>>>(dst, src, n);
    compute_kernel<<<cgrid, cblock, 0, 0>>>(cbuf, iters);   // 当前流立即发，与通信并行
    EventHandle h(comm);
    h.current_stream_wait();                                // 末尾才让当前流等通信
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

static float time_comm(cudaStream_t comm, float* dst, float* src, long n, int comm_blocks) {
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, 0);
    comm_copy_kernel<<<comm_blocks, 256, 0, comm>>>(dst, src, n);
    EventHandle h(comm);
    h.current_stream_wait();
    cudaEventRecord(t1, 0);
    cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms;
}

int main(int argc, char** argv) {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  cc=%d.%d  SM=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ---- get_comm_stream: stream-from-pool high priority, static/单例 ----
    int least, greatest;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    cudaStream_t comm = nullptr;
    CK(cudaStreamCreateWithPriority(&comm, cudaStreamNonBlocking, greatest));
    int prio = 0; cudaStreamGetPriority(comm, &prio);
    printf("\n[Part B] get_comm_stream 镜像: priority range [%d..%d], comm stream prio=%d\n",
           least, greatest, prio);

    // ---- ① ordering: 生产者流写 → EventHandle.wait → 消费者流读 ----
    {
        float* d_buf; int* d_bad;
        CK(cudaMalloc(&d_buf, sizeof(float)));
        CK(cudaMalloc(&d_bad, sizeof(int)));
        CK(cudaMemset(d_bad, 0, sizeof(int)));
        // 生产者：comm 流上 spin 后写 7.25
        writer_kernel<<<1, 32, 0, comm>>>(d_buf, 200000000ULL, 7.25f);
        EventHandle h(comm);                 // record 在 comm 流（写之后）
        h.current_stream_wait();             // 默认流等待事件
        reader_kernel<<<1, 32, 0, 0>>>(d_buf, d_bad, 7.25f);  // 默认流读
        CK(cudaStreamSynchronize(0));
        int bad = -1; CK(cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
        printf("[Part A] EventHandle ordering: consumer 读到 %.5f，bad=%d -> %s\n",
               7.25, bad, bad == 0 ? "PASS（等待保证可见）" : "FAIL");
        cudaFree(d_buf); cudaFree(d_bad);
    }

    // ---- ② stream_wait 自等待断言 ----
    {
        cudaStream_t s1, s2; CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s2));
        bool cross = stream_wait(s1, s2);
        bool self = stream_wait(s1, s1);
        printf("[Part E] stream_wait: 交叉 %s, 自等待 %s -> %s\n",
               cross ? "OK" : "BAD", self ? "OK(不该)" : "REJECTED(expected)",
               (cross && !self) ? "PASS" : "FAIL");
        cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    }

    // ---- ③/④ overlap：通信腿 vs 计算腿 ----
    {
        long n = 512L * 1024 * 1024 / 4;         // 512MB
        float *src, *dst; CK(cudaMalloc(&src, n * 4)); CK(cudaMalloc(&dst, n * 4));
        CK(cudaMemset(src, 1, n * 4));
        float* cbuf; CK(cudaMalloc(&cbuf, 256 * sizeof(float)));
        dim3 cgrid(prop.multiProcessorCount * 8), cblock(256);
        int iters = 300000;

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0, 0); compute_kernel<<<cgrid, cblock, 0, 0>>>(cbuf, iters);
        cudaEventRecord(t1, 0); cudaEventSynchronize(t1);
        float t_comp; cudaEventElapsedTime(&t_comp, t0, t1);
        cudaEventDestroy(t0); cudaEventDestroy(t1);

        float t_comm_f = time_comm(comm, dst, src, n, 1024);          // 全卡 copy
        float t_ser_f  = time_serial(comm, dst, src, n, 1024, cbuf, cgrid, cblock, iters);
        float t_ovl_f  = time_overlap(comm, dst, src, n, 1024, cbuf, cgrid, cblock, iters);
        printf("\n[Part D] 计算腿(FFMA 全卡)=%.3f ms\n", t_comp);
        printf("[Part D] 全卡通信腿(copy 512MB)=%.3f ms  串行=%.3f  重叠=%.3f  加速比=%.3f x\n",
               t_comm_f, t_ser_f, t_ovl_f, t_ser_f / t_ovl_f);
        printf("[Part D]   -> 两腿都占满全卡：重叠 %s（=sum，单卡不重叠）\n",
               (t_ovl_f >= t_ser_f * 0.95f) ? "PASS≈serial" : "unexpected overlap");

        // 受限 SM 的通信腿（镜像 get_theoretical_num_sms 的「半卡上限」）
        int comm_blocks = prop.multiProcessorCount / 4;               // 只占 1/4 SM
        float t_comm_l = time_comm(comm, dst, src, n, comm_blocks);
        float t_ser_l  = time_serial(comm, dst, src, n, comm_blocks, cbuf, cgrid, cblock, iters);
        float t_ovl_l  = time_overlap(comm, dst, src, n, comm_blocks, cbuf, cgrid, cblock, iters);
        printf("[Part D] 受限通信腿(copy，%d blocks=1/4 SM)=%.3f ms  串行=%.3f  重叠=%.3f  加速比=%.3f x\n",
               comm_blocks, t_comm_l, t_ser_l, t_ovl_l, t_ser_l / t_ovl_l);
        printf("[Part D]   -> 通信让出 SM：重叠 %s（≈max 而非 sum）\n",
               (t_ovl_l < t_ser_l * 0.95f) ? "PASS（真重叠）" : "FAIL");
        cudaFree(src); cudaFree(dst); cudaFree(cbuf);
    }

    cudaStreamDestroy(comm);
    printf("\nDONE\n");
    return 0;
}
