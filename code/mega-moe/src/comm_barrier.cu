// P77-1 实验：DeepGEMM 的**跨 SM / 跨 rank 同步原语** `comm/barrier.cuh`
//   （`wait_until` 超时自旋、`cluster_sync_with_relaxed_arrive`、`grid_sync`、`nvlink_barrier`）
//   与设备端**对称指针翻译** `layout/sym_buffer.cuh`，在 H100(sm90) 上**直接编译运行真实设备代码**。
//
// 背景：第 11 篇在 `comm/barrier.cuh` 的「原语表」里只列了 `grid_sync:33-51` / `nvlink_barrier:53-92`
//   两行；第 38 篇拆 PTX 层时把它们当 `ptx::` 的**调用点**引用（内存序表 `barrier.cuh:42/45/75/82`），
//   并明说「`grid_sync` 的整体语义因共享 GPU 会死锁，仅源码解读」。本篇把这层**单独**拉到真机上验证：
//   全是 `__device__` 内联、架构无关（不碰 UMMA/TMA），sm90 可编译可运行。
//
// 覆盖（全部 include 真实头文件，非手抄）：
//   A) `grid_sync`：单 rank 内全 SM barrier。验证 ① 它确实把「同步前所有 block 的写」发布给
//      「同步后所有 block 的读」；② `kFinishSumTag=0x80000000` 的**免复位**特性——计数器每轮
//      加 2^31（mod 2^32）故只在高位翻转、两轮回到原值，可无限复用；③ 多组 SM 数（含奇数 N）。
//   B) `nvlink_barrier` 的**状态机**（单卡虚拟 N 个 rank，`SymBuffer` offsets 全 0）：
//      验证 counter 的低两位如何编码 (phase,sign)、±1 信号如何在 4 次 barrier 内自清零。
//   C) `nvlink_barrier` **真实 8 卡**（UVA peer 对称缓冲）：8 rank 并发跑 R 次 barrier，
//      并验证 barrier 后能从对端读到本轮到齐的数据（`st_rel_sys`/`ld_acq_sys`）。
//   D) `cluster_sync_with_relaxed_arrive`：2-CTA cluster 内 relaxed arrive + wait，测其
//      「弱内存序」在真机上对**同步前写**的可见性（对照 grid_sync 的强序）。
//
// 编译/运行见 harness/exp_comm_barrier.sh。目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

#include <deep_gemm/comm/barrier.cuh>

using namespace deep_gemm;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(1);} } while (0)

// ---------------------------------------------------------------------------
// A) grid_sync
// ---------------------------------------------------------------------------
template <uint32_t kNumSMs>
__global__ void k_grid_sync(layout::Workspace ws, uint32_t* flags, uint32_t rounds,
                            uint32_t* out_mismatch, uint32_t* out_counters, uint32_t seq_len) {
    const uint32_t sm = blockIdx.x, tid = threadIdx.x;
    uint32_t mism = 0;
    for (uint32_t r = 0; r < rounds; ++ r) {
        // 每轮用**独立槽** flags[r*kNumSMs + sm]，避免「下一轮写踩上一轮读」的测试竞态
        if (tid == 0)
            flags[(size_t)r * kNumSMs + sm] = r + 1;
        // 全 SM barrier；sync_scope = __syncthreads（真实 kernel 里是 ptx::sync_aligned 的命名 barrier）
        comm::grid_sync<kNumSMs, 0>(ws, sm, tid, [=]() { __syncthreads(); });
        // barrier 之后读本轮所有格：应恰好都是 r+1
        if (tid == 0) {
            for (uint32_t b = 0; b < kNumSMs; ++ b)
                if (flags[(size_t)r * kNumSMs + b] != r + 1) ++ mism;
            if (sm == 0 and r < seq_len)
                out_counters[r] = ptx::ld_acq(ws.get_grid_sync_count_ptr<0>());
        }
    }
    if (tid == 0)
        out_mismatch[sm] = mism;
}

template <uint32_t kNumSMs>
static void part_a(uint32_t G, uint32_t rounds, uint32_t* d_signals) {
    layout::Workspace ws(d_signals, 1, 8, 1, 1, 8);

    uint32_t *d_flags, *d_mism, *d_seq;
    CK(cudaMalloc(&d_flags, (size_t)G * rounds * sizeof(uint32_t)));
    CK(cudaMalloc(&d_mism, G * sizeof(uint32_t)));
    constexpr uint32_t kSeq = 6;
    CK(cudaMalloc(&d_seq, kSeq * sizeof(uint32_t)));
    CK(cudaMemset(d_flags, 0, (size_t)G * rounds * sizeof(uint32_t)));
    CK(cudaMemset(d_mism, 0, G * sizeof(uint32_t)));
    CK(cudaMemset(d_seq, 0, kSeq * sizeof(uint32_t)));

    k_grid_sync<kNumSMs><<<G, 128>>>(ws, d_flags, rounds, d_mism, d_seq, kSeq);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> mism(G), seq(kSeq);
    CK(cudaMemcpy(mism.data(), d_mism, G * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(seq.data(), d_seq, kSeq * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    uint32_t total_mism = 0;
    for (auto v : mism) total_mism += v;

    printf("GRIDSYNC G=%u rounds=%u mismatch=%u counter_seq=%u,%u,%u,%u,%u,%u\n",
           G, rounds, total_mism, seq[0], seq[1], seq[2], seq[3], seq[4], seq[5]);
    CK(cudaFree(d_flags)); CK(cudaFree(d_mism)); CK(cudaFree(d_seq));
}

// ---------------------------------------------------------------------------
// B) nvlink_barrier 状态机（单卡虚拟 kNumRanks 个 rank，SymBuffer offsets 全 0）
//    signal 收集到同一地址：thread_idx<kNumRanks 各 add ±1，wait 等 target=±kNumRanks。
// ---------------------------------------------------------------------------
template <uint32_t kNumSMs, uint32_t kNumRanks>
__global__ void k_nvl_virtual(layout::Workspace ws, layout::SymBuffer<kNumRanks> sym,
                              uint32_t rounds, uint32_t* out_trace, uint32_t trace_len) {
    const uint32_t sm = blockIdx.x, tid = threadIdx.x;
    for (uint32_t r = 0; r < rounds; ++ r) {
        comm::nvlink_barrier<kNumRanks, kNumSMs, 256, 0, 7>(
            ws, sym, sm, tid, [=]() { __syncthreads(); });
        if (tid == 0 and sm == 0 and r < trace_len) {
            const uint32_t c = *ws.get_nvl_barrier_counter_ptr();
            out_trace[r * 3 + 0] = c & 3;                                   // status = counter & 3
            out_trace[r * 3 + 1] = (uint32_t)ptx::ld_acq_sys(ws.get_nvl_barrier_signal_ptr(0));
            out_trace[r * 3 + 2] = (uint32_t)ptx::ld_acq_sys(ws.get_nvl_barrier_signal_ptr(1));
        }
    }
}

template <uint32_t kNumSMs, uint32_t kNumRanks>
static void part_b(uint32_t G, uint32_t rounds, uint32_t* d_signals) {
    layout::Workspace ws(d_signals, 1, 8, 1, 1, 8);
    std::vector<int64_t> same(kNumRanks, reinterpret_cast<int64_t>(d_signals));
    layout::SymBuffer<kNumRanks> sym(same, 0);   // 所有 rank 映射到同一地址

    const uint32_t trace_len = 16;
    uint32_t* d_trace;
    CK(cudaMalloc(&d_trace, trace_len * 3 * sizeof(uint32_t)));
    CK(cudaMemset(d_trace, 0, trace_len * 3 * sizeof(uint32_t)));

    k_nvl_virtual<kNumSMs, kNumRanks><<<G, 256>>>(ws, sym, rounds, d_trace, trace_len);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> tr(trace_len * 3);
    CK(cudaMemcpy(tr.data(), d_trace, trace_len * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 期望：第 r 轮用 status=r&3 解出 (phase,sign)，向 signal[phase] target=sign?0:N 收敛；
    //   记录的是 barrier 之后的值，故 counter 已 +1（status_post=(r+1)&3）。
    int bad = 0;
    uint32_t e0 = 0, e1 = 0;
    for (uint32_t r = 0; r < trace_len; ++ r) {
        uint32_t status = r & 3, phase = status & 1, sign = status >> 1;
        if (phase == 0) e0 = sign ? 0 : kNumRanks;
        else            e1 = sign ? 0 : kNumRanks;
        if (tr[r*3+0] != ((r + 1) & 3) or tr[r*3+1] != e0 or tr[r*3+2] != e1) ++ bad;
    }
    printf("NVLVIrt N=%u G=%u rounds=%u trace_bad=%d\n", kNumRanks, G, rounds, bad);
    printf("NVLVIrt_trace (status,sig0,sig1):");
    for (uint32_t r = 0; r < 8; ++ r)
        printf(" (%u,%u,%u)", tr[r*3+0], tr[r*3+1], tr[r*3+2]);
    printf("\n");
    CK(cudaFree(d_trace));
}

// ---------------------------------------------------------------------------
// C) nvlink_barrier 真实多卡（UVA peer 对称缓冲）
// ---------------------------------------------------------------------------
template <uint32_t kNumSMs, uint32_t kNumRanks>
__global__ void k_nvl_real(layout::Workspace ws, layout::SymBuffer<kNumRanks> sym,
                           uint64_t* data, uint32_t rounds, uint64_t* out_mismatch) {
    const uint32_t sm = blockIdx.x, tid = threadIdx.x;
    uint64_t mism = 0;
    for (uint32_t r = 0; r < rounds; ++ r) {
        // 每轮独立槽 data[r*kNumRanks + rank]：本 rank 写自己那格，对端 barrier 后读
        if (tid == 0)
            ptx::st_rel_sys(data + (size_t)r * kNumRanks + sym.rank_idx, (uint64_t)r * 1000ull + sym.rank_idx);
        comm::nvlink_barrier<kNumRanks, kNumSMs, 256, 0, 11>(
            ws, sym, sm, tid, [=]() { __syncthreads(); });
        if (tid == 0) {
            for (uint32_t p = 0; p < kNumRanks; ++ p) {
                const uint64_t v = ptx::ld_acq_sys(sym.map(data + (size_t)r * kNumRanks + p, p));
                if (v != (uint64_t)r * 1000ull + p) ++ mism;
            }
        }
    }
    if (tid == 0 and sm == 0) *out_mismatch = mism;
}

// ---------------------------------------------------------------------------
// D) cluster_sync_with_relaxed_arrive（2-CTA cluster）
// ---------------------------------------------------------------------------
__global__ void __cluster_dims__(2, 1, 1) k_cluster(uint32_t* flags, uint32_t* out) {
    const uint32_t b = blockIdx.x, c = b & 1u;   // cluster 内 rank
    if (threadIdx.x == 0)
        flags[c] = 0xabc00000u + b;
    comm::cluster_sync_with_relaxed_arrive();
    if (threadIdx.x == 0) {
        const uint32_t other = flags[c ^ 1u];
        out[b] = (other == 0xabc00000u + (b ^ 1u)) ? 1u : 0u;
    }
}

// ---------------------------------------------------------------------------
// C) 真实多卡 host 编排：每卡一块等长对称缓冲（signals + data），UVA peer 互访。
// ---------------------------------------------------------------------------
template <uint32_t kNumSMs, uint32_t kNumRanks>
static void part_c(uint32_t G, uint32_t rounds) {
    static_assert(kNumRanks <= 8, "this host path supports up to 8 devices");

    const size_t sig_bytes = sizeof(layout::MegaMoESignals);
    const size_t data_off = (sig_bytes + 127) & ~size_t(127);
    const size_t buf_bytes = data_off + (size_t)rounds * kNumRanks * sizeof(uint64_t);

    std::vector<void*> bases(kNumRanks, nullptr);
    for (uint32_t i = 0; i < kNumRanks; ++ i) {
        CK(cudaSetDevice(i));
        CK(cudaMalloc(&bases[i], buf_bytes));
        CK(cudaMemset(bases[i], 0, buf_bytes));
    }
    // 打开两两 peer access（失败不致命——UVA 下 map 仍可能工作，这里要求成功）
    for (uint32_t i = 0; i < kNumRanks; ++ i) {
        CK(cudaSetDevice(i));
        for (uint32_t j = 0; j < kNumRanks; ++ j) {
            if (i == j) continue;
            int can = 0;
            CK(cudaDeviceCanAccessPeer(&can, i, j));
            if (can) { cudaError_t e = cudaDeviceEnablePeerAccess(j, 0); if (e != cudaSuccess and e != cudaErrorPeerAccessAlreadyEnabled) CK(e); }
        }
    }

    std::vector<int64_t> ptrs(kNumRanks);
    for (uint32_t i = 0; i < kNumRanks; ++ i) ptrs[i] = reinterpret_cast<int64_t>(bases[i]);

    std::vector<uint64_t*> d_mism(kNumRanks);
    for (uint32_t i = 0; i < kNumRanks; ++ i) {
        CK(cudaSetDevice(i));
        layout::Workspace ws(bases[i], kNumRanks, kNumRanks * 8, 1, 1, 8);
        layout::SymBuffer<kNumRanks> sym(ptrs, i);
        uint64_t* data = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(bases[i]) + data_off);
        CK(cudaMalloc(&d_mism[i], sizeof(uint64_t)));
        CK(cudaMemset(d_mism[i], 0, sizeof(uint64_t)));
        k_nvl_real<kNumSMs, kNumRanks><<<G, 256>>>(ws, sym, data, rounds, d_mism[i]);
        CK(cudaGetLastError());
    }
    for (uint32_t i = 0; i < kNumRanks; ++ i) { CK(cudaSetDevice(i)); CK(cudaDeviceSynchronize()); }

    uint64_t total_mism = 0;
    for (uint32_t i = 0; i < kNumRanks; ++ i) {
        CK(cudaSetDevice(i));
        uint64_t m = 0; CK(cudaMemcpy(&m, d_mism[i], sizeof(uint64_t), cudaMemcpyDeviceToHost));
        total_mism += m;
        CK(cudaFree(d_mism[i]));
        CK(cudaFree(bases[i]));
        cudaDeviceDisablePeerAccess(0);  // 忽略错误
    }
    printf("NVLRANK ranks=%u G=%u rounds=%u mismatch=%llu\n",
           kNumRanks, G, rounds, (unsigned long long)total_mism);
}

// ---------------------------------------------------------------------------
// D) cluster_sync_with_relaxed_arrive
// ---------------------------------------------------------------------------
static void part_d() {
    uint32_t *d_flags, *d_out;
    CK(cudaMalloc(&d_flags, 2 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_out, 2 * sizeof(uint32_t)));
    CK(cudaMemset(d_flags, 0, 2 * sizeof(uint32_t)));
    CK(cudaMemset(d_out, 0, 2 * sizeof(uint32_t)));
    k_cluster<<<2, 32>>>(d_flags, d_out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    uint32_t out[2];
    CK(cudaMemcpy(out, d_out, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("CLUSTER dim=2 seen_peer=%u/%u\n", out[0], out[1]);
    CK(cudaFree(d_flags)); CK(cudaFree(d_out));
}

int main() {
    int dev = 0, num_sms = 0, ndev = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
    CK(cudaGetDeviceCount(&ndev));
    printf("Device sm_count=%d device_count=%d\n", num_sms, ndev);

    // 复用一块信号缓冲（MegaMoESignals 大小），清零
    void* sig = nullptr;
    CK(cudaMalloc(&sig, sizeof(layout::MegaMoESignals)));
    CK(cudaMemset(sig, 0, sizeof(layout::MegaMoESignals)));
    uint32_t* d_signals = reinterpret_cast<uint32_t*>(sig);

    const uint32_t kRounds = 2000;
    uint32_t* d_signals_a = d_signals;

    // ---- Part A ----
    part_a<1>(1, kRounds, d_signals_a);
    part_a<2>(2, kRounds, d_signals_a);
    part_a<8>(8, kRounds, d_signals_a);
    part_a<16>(16, kRounds, d_signals_a);
    part_a<64>(64, kRounds, d_signals_a);
    part_a<7>(7, kRounds, d_signals_a);      // 奇数 N（走 sm_idx==0 的 tag-(N-1) 分支）

    // ---- Part B ----
    CK(cudaMemset(sig, 0, sizeof(layout::MegaMoESignals)));
    const uint32_t kRoundsB = 1000;
    part_b<8, 2>(8, kRoundsB, d_signals_a);
    CK(cudaMemset(sig, 0, sizeof(layout::MegaMoESignals)));
    part_b<8, 8>(8, kRoundsB, d_signals_a);

    CK(cudaFree(sig));
    CK(cudaDeviceSynchronize());
    printf("SINGLE_GPU_DONE\n");

    // ---- Part D ----
    part_d();

    // ---- Part C (真实 8 卡) ----
    if (ndev >= 2) {
        const uint32_t ranks = ndev >= 8 ? 8u : (uint32_t)ndev;
        switch (ranks) {
            case 2: part_c<16, 2>(16, 30); break;
            case 4: part_c<16, 4>(16, 30); break;
            case 8: part_c<16, 8>(16, 30); break;
            default: printf("NVLRANK skipped (ranks=%u)\n", ranks);
        }
    }
    printf("ALL_DONE\n");
    return 0;
}
