// P45-1 实验：DeepGEMM 的 per-warp `%clock64` 追踪探针
//   （`deep_gemm/include/deep_gemm/common/probe.cuh`，公开版**无任何调用点**的基础设施）
//   在 H100(sm90) 上**直接编译运行真实设备代码**。
//
// 背景：前面 0~44 篇多次依赖 `clock64` 做计时/标定（P24 局部域 pointer-chase 探针、P26
//   `set_tc_util` 的 tensor-core 节流空转、P34/P52 的时序实验），`common/probe.cuh` 则是
//   DeepGEMM 自己提供的**通用 per-warp trace 设施**：把 `%clock64` 写成一串定长 8 字节记录到
//   全局内存，两端用 `mark()`（`clock64` + `%globaltimer`）做时钟标定。它默认**编译为空**
//   （`DG_PROBE_ENABLE` 未定义时 `=0`），JIT 也从不定义它，所以公开仓库里它是一段**休眠的
//   观测基础设施**——本实验把它拉到 H100 真机上逐条验证语义与开销。
//
// 全部为架构无关的设备内联 asm（`mov %clock64` / `mov %globaltimer` / `st.global`），
// sm90 可编译可运行。覆盖：
//   A) `DG_PROBE_ENABLE` 开关语义：`kEnabled` / 关闭后编成空（缓冲区保持全零）。
//   B) `record<offset>` / `advance<N>` / `record_payload` / `mark` 的记录顺序与偏移语义
//      （含「先 advance、再以负偏移回填」的约定）。
//   C) `mark()` 用 `%globaltimer`（ns）标定 SM 时钟：由 `clock64/globaltimer` 算出 SM 频率。
//   D) `is_valid_buffer` / `get_num_bytes` 的 host 侧安全判据（对齐 / 记录数偶数 /
//      **必须落在同一个 4 GiB 窗口内**）。
//   E) `advance<N>` **只加低 32 位**（单条 `IADD3`）——用一个伪指针坐实「高 32 位永不进位」，
//      这正是 D 里 4 GiB 窗口约束的根因。
//   F) 探针开销：同一循环 opens/closes probe 的墙时间差 → 每条记录 ~ns 级。
//
// 编译/运行见 harness/exp_probe_clock64.sh；raw 二进制写 /tmp/probe_*.bin，判定在同样脚本里的
// Python 段（用 numpy 解析记录流）。目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

// 真实设备代码（原样 include；`DG_PROBE_ENABLE` 由命令行控制）
#include <deep_gemm/common/probe.cuh>

using namespace deep_gemm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

static void dump(const char* path, const void* ptr, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) { printf("cannot open %s\n", path); std::exit(1); }
    fwrite(ptr, 1, bytes, f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// A) 开关语义：`kEnabled = DG_PROBE_ENABLE and kNumRecords > 0`
// ---------------------------------------------------------------------------
__global__ void k_enabled(int* out) {
    out[0] = (int)probe::Stream<8>::kEnabled;
    out[1] = (int)probe::Stream<0>::kEnabled;   // kNumRecords == 0 → 恒关
    out[2] = (int)probe::Stream<8>::kNumBytesPerRecord;
    out[3] = (int)probe::Stream<8>::kNumBytes;
    out[4] = (int)probe::Stream<8>::get_num_bytes(3);
}

// ---------------------------------------------------------------------------
// B) trace：记录顺序与偏移语义（单线程）
//    mark()      : r0=clock64, r1=globaltimer, cursor+=2
//    record<0>() : 写 cursor+0        （不前进）
//    advance<1>() 后 record<0>() : 写下一槽
//    advance<2>() 后 record<-2>/<-1> : 回填刚跳过的两槽（「先 advance 再负偏移」约定）
//    record_payload<0>()             : 写 16 字节（两槽）
//    mark()      : 末尾标定
// ---------------------------------------------------------------------------
__global__ void k_trace(void* buf) {
    if (threadIdx.x != 0 or blockIdx.x != 0) return;
    probe::Stream<64> s(buf, 0);

    s.mark();                                        // r0, r1   cursor -> 2
    s.record<0>();                                   // r2       cursor 2
    s.advance<1>();                                  //          cursor -> 3
    s.record<0>();                                   // r3       cursor 3
    s.advance<1>();                                  //          cursor -> 4
    s.advance<2>();                                  //          cursor -> 6
    s.record<-2>();                                  // r4 (回填刚跳过的槽)
    s.record<-1>();                                  // r5 (回填刚跳过的槽)
    // cursor 仍在槽 6（16 字节对齐），payload 才能用 v4.b32
    s.record_payload<0>(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);  // r6, r7
    s.advance<2>();                                  //          cursor -> 8
    __nanosleep(20000);                              // ~20us，让 clock64 明显前进
    s.record<0>();                                   // r8       cursor 8
    s.advance<1>();                                  //          cursor -> 9
    s.mark();                                        // r9, r10  cursor -> 11
}

// ---------------------------------------------------------------------------
// C) 时钟标定：`clock64`（SM 周期）对 `%globaltimer`（ns）
// ---------------------------------------------------------------------------
__global__ void k_calib(uint64_t* out) {
    uint64_t c0, c1, g0, g1;
    asm volatile("mov.u64 %0, %%clock64;"    : "=l"(c0));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    // 用 clock64 自旋烧掉固定周期数（不可被优化，也不依赖挂钟）
    uint64_t now = c0;
    const uint64_t kBurnCycles = 20000000ULL;
    while (now - c0 < kBurnCycles)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(now));
    asm volatile("mov.u64 %0, %%clock64;"    : "=l"(c1));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    out[0] = c0; out[1] = c1; out[2] = g0; out[3] = g1;
}

// ---------------------------------------------------------------------------
// E) advance 只加低 32 位：用伪指针构造 Stream（advance 不解引用）
//    pointer 高 32 位被 `Stream` ctor 原样放进 cursor，advance 只用 IADD3 加低 32 位。
// ---------------------------------------------------------------------------
__global__ void k_advance_low32(uint64_t* out) {
    {
        probe::Stream<2> s(reinterpret_cast<void*>(0x00000001FFFFFFF0ULL), 0);
        s.advance<2>();                              // +16 字节，低 32 位回绕
        out[0] = s.cursor;
    }
    {
        probe::Stream<2> s(reinterpret_cast<void*>(0x00000000FFFFFFF0ULL), 0);
        s.advance<2>();
        out[1] = s.cursor;
    }
    {
        probe::Stream<2> s(reinterpret_cast<void*>(0x0000FFF300000000ULL), 0);
        s.advance<1>();                              // +8
        out[2] = s.cursor;
    }
}

// ---------------------------------------------------------------------------
// F) 开销：同一循环，开/关 probe
// ---------------------------------------------------------------------------
__global__ void k_overhead(void* buf, int iters, int work, float* sink) {
    probe::Stream<262144> s(buf, 0);
    float acc = 0.0f;
    for (int i = 0; i < iters; ++i) {
        for (int j = 0; j < work; ++j) acc = fmaf(acc, 1.0000001f, 0.5f);
        s.record<0>();
        s.advance<1>();
    }
    if (acc == 123456.0f) *sink = acc;               // 防优化
}

// 单独量记录吞吐：每个 warp 一条独立 stream
__global__ void k_probe_only(void* buf, int iters) {
    const uint32_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    probe::Stream<4096> s(buf, warp);
    for (int i = 0; i < iters; ++i) {
        s.record<0>();
        s.advance<1>();
    }
}

int main(int argc, char** argv) {
    const char* tag = (argc > 1) ? argv[1] : "x";
    char path[256];
    int* d_flag; CK(cudaMalloc(&d_flag, 8 * sizeof(int)));
    k_enabled<<<1, 1>>>(d_flag);
    CK(cudaDeviceSynchronize());
    int flag[8]; CK(cudaMemcpy(flag, d_flag, 8 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("kEnabled<8>=%d kEnabled<0>=%d kNumBytesPerRecord=%d kNumBytes=%d get_num_bytes(3)=%d\n",
           flag[0], flag[1], flag[2], flag[3], flag[4]);

    // B) trace
    const uint32_t kRec = 64;
    char* h_buf = (char*)calloc(kRec * 8, 1);
    void* d_buf; CK(cudaMalloc(&d_buf, kRec * 8));
    CK(cudaMemset(d_buf, 0, kRec * 8));
    k_trace<<<1, 1>>>(d_buf);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h_buf, d_buf, kRec * 8, cudaMemcpyDeviceToHost));
    snprintf(path, sizeof(path), "/tmp/probe_%s_trace.bin", tag);
    dump(path, h_buf, kRec * 8);
    printf("trace written (%u records)\n", kRec);
    free(h_buf);

    // C) calibration
    uint64_t* d_cal; CK(cudaMalloc(&d_cal, 4 * sizeof(uint64_t)));
    k_calib<<<1, 1>>>(d_cal);
    CK(cudaDeviceSynchronize());
    uint64_t cal[4]; CK(cudaMemcpy(cal, d_cal, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("calib: c0=%llu c1=%llu g0=%llu g1=%llu\n",
           (unsigned long long)cal[0], (unsigned long long)cal[1],
           (unsigned long long)cal[2], (unsigned long long)cal[3]);
    snprintf(path, sizeof(path), "/tmp/probe_%s_calib.bin", tag);
    dump(path, cal, sizeof(cal));

    // E) advance low-32
    uint64_t* d_adv; CK(cudaMalloc(&d_adv, 3 * sizeof(uint64_t)));
    k_advance_low32<<<1, 1>>>(d_adv);
    CK(cudaDeviceSynchronize());
    uint64_t adv[3]; CK(cudaMemcpy(adv, d_adv, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("advance: [0x%016llx] [0x%016llx] [0x%016llx]\n",
           (unsigned long long)adv[0], (unsigned long long)adv[1], (unsigned long long)adv[2]);
    snprintf(path, sizeof(path), "/tmp/probe_%s_advance.bin", tag);
    dump(path, adv, sizeof(adv));

    // D) host 侧判据
    {
        snprintf(path, sizeof(path), "/tmp/probe_%s_valid.txt", tag);
        FILE* f = fopen(path, "w");
        // 用真实（已分配）指针做安全基线
        uint64_t base = (uint64_t)d_buf;                          // cudaMalloc 256 对齐
        uint64_t aligned = (base + 15) & ~15ULL;
        auto report = [&](const char* name, bool v) { fprintf(f, "%s %d\n", name, (int)v); };
        report("real_aligned", probe::Stream<64>::is_valid_buffer((void*)aligned, 1));
        report("real_misaligned8", probe::Stream<64>::is_valid_buffer((void*)(aligned + 8), 1));
        // 记录数奇偶：用 <63>（奇数）→ is_valid_buffer 恒 false
        report("odd_records", probe::Stream<63>::is_valid_buffer((void*)aligned, 1));
        // 4GiB 窗口：base 距 4GiB 边界只有 256 字节，64 条记录 = 512B 会越界（纯算术，不解引用）
        report("crosses_4gib", probe::Stream<64>::is_valid_buffer((void*)0x00000000FFFFFFF0ULL, 1));
        report("near_but_inside", probe::Stream<64>::is_valid_buffer((void*)(0x00000000FFFFFFF0ULL - 512), 1));
        // 完全落在窗口内（用假的低地址）
        report("inside_window", probe::Stream<64>::is_valid_buffer((void*)0x0000000000010000ULL, 1));
        // 多流：3 条流 1536B 仍落在同一窗口
        report("multi_stream_stay", probe::Stream<64>::is_valid_buffer((void*)aligned, 3));
        fprintf(f, "num_bytes_1 %llu\n", (unsigned long long)probe::Stream<64>::get_num_bytes(1));
        fprintf(f, "num_bytes_3 %llu\n", (unsigned long long)probe::Stream<64>::get_num_bytes(3));
        fclose(f);
    }

    // F) 开销
    {
        const int iters = 200000, work = 20;
        float* d_sink; CK(cudaMalloc(&d_sink, sizeof(float)));
        void* d_obuf; CK(cudaMalloc(&d_obuf, probe::Stream<262144>::kNumBytes));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        auto timeit = [&]() {
            k_overhead<<<1, 1>>>(d_obuf, iters, work, d_sink);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int r = 0; r < 20; ++r) k_overhead<<<1, 1>>>(d_obuf, iters, work, d_sink);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
            return ms / 20.0f;
        };
        float ms = timeit();
        printf("overhead: %.4f ms/kernel (iters=%d work=%d)\n", ms, iters, work);
        snprintf(path, sizeof(path), "/tmp/probe_%s_overhead.txt", tag);
        FILE* f = fopen(path, "w");
        fprintf(f, "%.6f %d %d\n", ms, iters, work);
        // 纯记录吞吐：4 warp × 4096 条
        const int piters = 4096, pwarps = 4;
        void* d_pbuf; CK(cudaMalloc(&d_pbuf, probe::Stream<4096>::kNumBytes * pwarps));
        auto timep = [&]() {
            k_probe_only<<<1, pwarps * 32>>>(d_pbuf, piters);
            CK(cudaDeviceSynchronize());
            CK(cudaEventRecord(e0));
            for (int r = 0; r < 50; ++r) k_probe_only<<<1, pwarps * 32>>>(d_pbuf, piters);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float t; CK(cudaEventElapsedTime(&t, e0, e1));
            return t / 50.0f;
        };
        float pms = timep();
        printf("probe_only: %.4f ms/kernel (%d warps x %d records)\n", pms, pwarps, piters);
        fprintf(f, "%.6f %d %d\n", pms, pwarps, piters);
        fclose(f);
        cudaFree(d_sink); cudaFree(d_obuf); cudaFree(d_pbuf);
    }

    CK(cudaFree(d_flag)); CK(cudaFree(d_buf)); CK(cudaFree(d_cal)); CK(cudaFree(d_adv));
    printf("done\n");
    return 0;
}
