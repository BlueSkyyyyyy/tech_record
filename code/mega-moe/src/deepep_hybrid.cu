// P61-1 实验：DeepEP 的两级（hybrid = scale-out RDMA × scale-up NVLink）EP
// dispatch / combine 的架构无关部分。
//
// 为什么 H100(sm90) 可跑：
//   * `impls/ep/hybrid_dispatch.cuh` / `hybrid_combine.cuh` 的**数据面**（TMA 搬运、
//     `gin.put/get_sym_ptr`、跨 rank barrier）走 NCCL 设备 API，本机 NCCL 2.29.7 不提供
//     （见 ROADMAP「阻塞」），故**仅源码分析**。
//   * 但它们的**计数/同步/调度内核**只踩公共设备层：
//       - `common/math.cuh` 的 `encode_decode_positive` / `is_decoded_positive_ready` /
//         `pack2` / `unpack2`（纯位运算，`__host__ __device__` 或 `__device__`）；
//       - `common/ptx.cuh` 的 `gather` / `ffs` / `reduce_or` / `exchange` /
//         `warp_inclusive_sum`（warp 内建）；
//       - `layout/ep/token.cuh` 的 `TokenLayout` 字节几何（纯结构体算术）。
//   本文件**直接 include 真实头文件、编译运行真实设备代码**：
//     Part 1  计数「就绪哨兵」编码 `encode_decode_positive`（`hybrid_dispatch.cuh:155-216`
//             的 notify 汇聚与 `recv_and_reduce` 全靠它区分「未写」与「写了 0」）；
//     Part 2  tail 信号 `pack2<int64>(finish, tail)` / `unpack2`（`hybrid_dispatch.cuh:341-343`
//             / `:519`、`hybrid_combine.cuh:590-596`）；
//     Part 3  `do_psum`（`hybrid_dispatch.cuh:304-320`，rank 含端点 / expert 排他前缀和）；
//     Part 4  forward warp 的 round-robin 分片调度（`hybrid_dispatch.cuh:490-532`）。
//   这些都不碰 NCCL、不碰 TMA，sm90 上编译运行真实设备代码，再用独立 host oracle 对拍。
//
//   nvcc -arch=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr -w \
//     -I~/github/DeepEP/deep_ep/include -I/usr/local/cuda/include/cccl \
//     src/deepep_hybrid.cu -o /tmp/deepep_hybrid && /tmp/deepep_hybrid

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/layout/ep/token.cuh>

using namespace deep_ep;
using namespace deep_ep::math;

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    printf("  [CUDA ERROR] %s (line %d): %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1);} }while(0)

static int g_pass = 0, g_fail = 0;
static void check(const bool ok, const std::string& what) {
    if (ok) { ++g_pass; printf("  [PASS] %s\n", what.c_str()); }
    else    { ++g_fail; printf("  [FAIL] %s\n", what.c_str()); }
}

// -----------------------------------------------------------------------------
// Part 1: 计数就绪哨兵 —— `encode_decode_positive` / `is_decoded_positive_ready`
//   math.cuh:26-33:
//     is_decoded_positive_ready(v) = (v >= 0)          // :26-28
//     encode_decode_positive(v)    = -v - 1            // :31-33
//   关键：把非负计数 c 编码成负值 (-c-1)，于是 buffer「初值 0」解码后 = -1（<0，未就绪），
//   而任何写过的 c（含 c=0，编码 -1，负值）解码后 = c（>=0，就绪）。一次对合即可把
//   「未写」与「写了 0」分开——注意**就绪判定作用在解码后的值上**（dispatch.cuh:200-201
//   先 decode 再 is_decoded_positive_ready），这是 notify 汇聚的同步基石。
// -----------------------------------------------------------------------------
__global__ void k_sentinel(const int* counts, int n, int* enc, int* dec, int* ready_dec, int* ready_unwritten) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = counts[i];
    const int e = encode_decode_positive<int>(c);      // 真实设备函数：写进 buffer 的负值
    enc[i] = e;
    const int d = encode_decode_positive<int>(e);      // 读回后 decode（对合 = 还原 c）
    dec[i] = d;
    ready_dec[i] = static_cast<int>(is_decoded_positive_ready<int>(d));               // 写过的 c>=0 → 就绪
    ready_unwritten[i] = static_cast<int>(is_decoded_positive_ready<int>(encode_decode_positive<int>(0))); // 初值 0 → decode=-1 → 未就绪
}

static void part1_sentinel() {
    printf("\n=== Part 1: 计数就绪哨兵 encode_decode_positive（math.cuh:26-33） ===\n");
    const int n = 8;
    std::vector<int> counts = {0, 1, 2, 7, 1023, 65535, 1 << 20, 0x7fffffff};
    int *d_counts, *d_enc, *d_dec, *d_re, *d_rz;
    CK(cudaMalloc(&d_counts, n * sizeof(int)));
    CK(cudaMalloc(&d_enc, n * sizeof(int)));
    CK(cudaMalloc(&d_dec, n * sizeof(int)));
    CK(cudaMalloc(&d_re, n * sizeof(int)));
    CK(cudaMalloc(&d_rz, n * sizeof(int)));
    CK(cudaMemcpy(d_counts, counts.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    k_sentinel<<<1, n>>>(d_counts, n, d_enc, d_dec, d_re, d_rz);
    CK(cudaDeviceSynchronize());
    std::vector<int> enc(n), dec(n), re(n), rz(n);
    CK(cudaMemcpy(enc.data(), d_enc, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dec.data(), d_dec, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(re.data(), d_re, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rz.data(), d_rz, n * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; ++ i) {
        const int expect_enc = -counts[i] - 1;
        const bool good = (enc[i] == expect_enc) and (dec[i] == counts[i]) and
                          (re[i] == 1) and (rz[i] == 0);
        ok &= good;
        printf("    c=%-11d raw(encode)=%-12d decode=%-11d ready(decode)=%d ready(unwritten0)=%d %s\n",
               counts[i], enc[i], dec[i], re[i], rz[i], good ? "" : "<<< BAD");
    }
    check(ok, "encode/decode 对合 + 哨兵语义（写过的 c 解码后就绪、初值 0 解码后未就绪）");
    // host 侧同一函数也要能编（__host__ __device__）
    bool host_ok = true;
    for (int c = 0; c < 4096; ++ c) {
        const int raw = encode_decode_positive<int>(c);            // 写进 buffer 的 -c-1
        const int decoded = encode_decode_positive<int>(raw);      // 读回后 decode = c
        host_ok &= (decoded == c) and is_decoded_positive_ready<int>(decoded) and
                   (not is_decoded_positive_ready<int>(encode_decode_positive<int>(0)));
    }
    check(host_ok, "同一函数 host 侧对合（4096 个计数）");
    cudaFree(d_counts); cudaFree(d_enc); cudaFree(d_dec); cudaFree(d_re); cudaFree(d_rz);
}

// -----------------------------------------------------------------------------
// Part 2: tail 信号 —— `pack2<int,int64_t>` / `unpack2`（math.cuh:45-68）
//   dispatch 的 `signaled_tail = pack2<int64>(finish_flag, stored_tail)`（:341）
//   并累加「新旧差」到对端（:347）；combine 用 `expected = pack2<int64>(1, 0)`（:590）
//   一次 red_add 后比等。`pack2<int,int64>` 把 x 放**低 32 位**、y 放**高 32 位**
//   （math.cuh:48-51: `int64` 重解释成 `int[2]`，`ptr[0]=x, ptr[1]=y`），故
//   `pack2(finish, tail)` = `(tail<<32) | finish`；`unpack2(p, f, t)` 对称解出
//   `f=p&0xffffffff, t=p>>32`（对应 combine:519 的 `unpack2(sig, finish, tail)`）。
//   验证：位段逐位一致 + 「只发增量」的多轮累加 = 最终 tail，finish 位独立。
// -----------------------------------------------------------------------------
__global__ void k_pack(const int* tails, const int* finishes, int n, long long* packed,
                       int* up_tail, int* up_finish) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto p = pack2<int, long long>(finishes[i], tails[i]);   // (finish=低, tail=高)
    packed[i] = p;
    int f, t;
    unpack2<int, long long>(p, f, t);                              // 真实设备解包
    up_tail[i] = t;
    up_finish[i] = f;
}

static void part2_pack() {
    printf("\n=== Part 2: tail 信号 pack2/unpack2（math.cuh:45-68；dispatch:341/347、combine:590-596） ===\n");
    const int n = 6;
    std::vector<int> tails = {0, 1, 6, 100, 1 << 20, 0x7fffffff};
    std::vector<int> fin = {0, 1, 1, 0, 1, 0};
    long long *d_p; int *d_ut, *d_uf, *d_t, *d_f;
    CK(cudaMalloc(&d_p, n * sizeof(long long)));
    CK(cudaMalloc(&d_ut, n * sizeof(int)));
    CK(cudaMalloc(&d_uf, n * sizeof(int)));
    CK(cudaMalloc(&d_t, n * sizeof(int)));
    CK(cudaMalloc(&d_f, n * sizeof(int)));
    CK(cudaMemcpy(d_t, tails.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_f, fin.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    k_pack<<<1, n>>>(d_t, d_f, n, d_p, d_ut, d_uf);
    CK(cudaDeviceSynchronize());
    std::vector<long long> p(n); std::vector<int> ut(n), uf(n);
    CK(cudaMemcpy(p.data(), d_p, n * sizeof(long long), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ut.data(), d_ut, n * sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(uf.data(), d_uf, n * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < n; ++ i) {
        const long long expect = (static_cast<long long>(tails[i]) << 32) | (static_cast<unsigned>(fin[i]));
        const bool good = (p[i] == expect) and (ut[i] == tails[i]) and (uf[i] == fin[i]);
        ok &= good;
        printf("    finish=%-2d tail=%-11d packed=0x%016llx unpack=(finish=%d,tail=%d) %s\n",
               fin[i], tails[i], static_cast<unsigned long long>(p[i]), uf[i], ut[i], good ? "" : "<<< BAD");
    }
    check(ok, "pack2/unpack2 低 32=finish、高 32=tail 逐位一致");

    // 「只发增量」的多轮累加：intervals {2,5,9} 的差值 2,3,4 → 最终 9；finish 由末轮置 1。
    // `pack2/unpack2` 是 `__device__`（math.cuh:44），host 侧用等价位运算复刻同一语义。
    const std::vector<int> rounds = {2, 5, 9};
    auto host_pack2 = [](int f, int t) -> long long {
        return (static_cast<long long>(t) << 32) | static_cast<unsigned>(f);   // finish 低、tail 高
    };
    auto host_unpack2 = [](long long p, int& f, int& t) {
        f = static_cast<int>(static_cast<unsigned>(p & 0xffffffffll));                     // finish 低
        t = static_cast<int>(static_cast<unsigned long long>(p) >> 32);                    // tail 高
    };
    long long acc = 0; int old = 0;
    for (size_t i = 0; i < rounds.size(); ++ i) {
        const bool finish = (i + 1 == rounds.size());
        const auto sig = host_pack2(finish ? 1 : 0, rounds[i]);
        const auto old_sig = host_pack2(0, old);
        acc += sig - old_sig;              // 对端 red_add(acc, sig - old_sig)
        old = rounds[i];
    }
    int f_final, t_final;
    host_unpack2(acc, f_final, t_final);
    check((t_final == 9) and (f_final == 1), "增量累加后 tail=9、finish=1（= dispatch:347 / combine:594）");
    cudaFree(d_p); cudaFree(d_ut); cudaFree(d_uf); cudaFree(d_t); cudaFree(d_f);
}

// -----------------------------------------------------------------------------
// Part 3: `do_psum` —— hybrid_dispatch.cuh:304-320 的逐字算法
//   warp0 对 scaleup rank 计数做**含端点**前缀和（往 psum_num_recv_tokens_per_scaleup_rank），
//   warp1 对 expert 计数做**排他**前缀和（往 psum_num_recv_tokens_per_expert，为展开铺位）。
//   这里逐字搬运算法（含 psum 进位 + `exchange(sum, 31)`），用 host oracle 对拍。
// -----------------------------------------------------------------------------
__device__ __forceinline__ void do_psum(const int* count, int* out, const int n, const int is_exclusive) {
    // 逐字来自 deep_ep/include/deep_ep/impls/ep/hybrid_dispatch.cuh:304-320
    const auto lane_idx = ptx::get_lane_idx();
    int psum = 0;
    #pragma unroll
    for (int i = 0; i < ceil_div(n + is_exclusive, 32); ++ i) {
        const auto idx = i * 32 + lane_idx;
        const auto mem_idx = idx - is_exclusive;
        const auto value = (0 <= mem_idx and mem_idx < n) ? count[mem_idx] : 0;
        const auto sum = psum + ptx::warp_inclusive_sum(value, lane_idx);
        if (idx < n + is_exclusive)
            out[idx] = sum;
        psum = ptx::exchange(sum, 31);
    }
}

__global__ void k_psum(const int* count, int* out, int n, int is_exclusive) {
    do_psum(count, out, n, is_exclusive);
}

static void part3_psum() {
    printf("\n=== Part 3: do_psum 前缀和（hybrid_dispatch.cuh:304-320 逐字搬运） ===\n");
    struct Case { int n; int excl; const char* name; };
    std::vector<Case> cases = {
        {1, 0, "n=1 inclusive"}, {32, 0, "n=32 inclusive"}, {33, 0, "n=33 inclusive"},
        {8, 1, "n=8 exclusive"}, {32, 1, "n=32 exclusive"}, {33, 1, "n=33 exclusive"},
        {70, 1, "n=70 exclusive"}, {130, 1, "n=130 exclusive"},
    };
    bool ok = true;
    for (auto& c : cases) {
        const int maxn = c.n;
        std::vector<int> count(std::max(maxn, 1));
        for (int i = 0; i < maxn; ++ i) count[i] = (i * 7 + 3) % 11;
        const int out_n = c.n + c.excl;
        std::vector<int> out(std::max(out_n, 1), -999);
        int *d_c, *d_o;
        CK(cudaMalloc(&d_c, std::max(maxn, 1) * sizeof(int)));
        CK(cudaMalloc(&d_o, std::max(out_n, 1) * sizeof(int)));
        CK(cudaMemcpy(d_c, count.data(), maxn * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_o, out.data(), std::max(out_n, 1) * sizeof(int), cudaMemcpyHostToDevice));
        k_psum<<<1, 32>>>(d_c, d_o, c.n, c.excl);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), d_o, out_n * sizeof(int), cudaMemcpyDeviceToHost));
        // host oracle
        std::vector<int> ref(out_n, 0);
        long long run = 0;
        if (c.excl) {
            ref[0] = 0; run = 0;
            for (int i = 0; i < c.n; ++ i) { run += count[i]; ref[i + 1] = (int)run; }
        } else {
            run = 0;
            for (int i = 0; i < c.n; ++ i) { run += count[i]; ref[i] = (int)run; }
        }
        bool good = true;
        for (int i = 0; i < out_n; ++ i) good &= (out[i] == ref[i]);
        ok &= good;
        printf("    %-16s count[0..2]={%d,%d,%d} out tail=%d ref=%d %s\n",
               c.name, count[0], count[1], count[2], out[out_n - 1], ref[out_n - 1], good ? "" : "<<< BAD");
        cudaFree(d_c); cudaFree(d_o);
    }
    check(ok, "do_psum 含端点(rank)/排他(expert) 前缀和 vs host oracle");
}

// -----------------------------------------------------------------------------
// Part 4: forward warp 的 round-robin 分片调度（hybrid_dispatch.cuh:490-532）
//   每个 lane 持有自己 rank 的已到达 tail（`stored_scaleout_tail_idx`，来自 :517-519 的
//   `ld_acquire_sys` + `unpack2`），用 `gather(tail > old)` 得 wip 位图，然后
//   `offset=(recv+1)%R; hi=wip>>offset<<offset; recv=hi?ffs(hi):ffs(wip)`
//   取「上一个处理过的 rank 之后、第一个有数据的 rank」；每步只处理一个 chunk
//   `[start, min(tail, start+kNumSlotsPerForwardChunk))`。
//   验证：每个 rank 的 [0, tail) 恰好被处理一次 + 选择序列 = 贪心 round-robin。
// -----------------------------------------------------------------------------
__global__ void k_rr(const int* tail_in, int R, int chunk, int* seq, int* seq_len, int* proc_count) {
    const int lane = ptx::get_lane_idx();
    int stored_tail = lane < R ? tail_in[lane] : 0;
    int stored_old = 0;
    int recv = 0;                       // uniform（所有 lane 同步）
    if (lane == 0) *seq_len = 0;
    if (lane < R) proc_count[lane] = 0;
    __syncwarp();
    int n = 0;
    unsigned wip;
    while ((wip = ptx::gather(stored_tail > stored_old)) != 0) {
        const int offset = (recv + 1) % R;
        const unsigned hi = (wip >> offset) << offset;
        recv = hi ? ptx::ffs(hi) : ptx::ffs(wip);
        const int start = ptx::exchange(stored_old, recv);
        const int end = min(ptx::exchange(stored_tail, recv), start + chunk);
        if (lane == recv) stored_old = end;
        if (lane == recv) proc_count[recv] += (end - start);
        if (lane == 0) seq[n] = recv;
        ++ n;
    }
    if (lane == 0) *seq_len = n;
}

static void part4_rr() {
    printf("\n=== Part 4: forward warp round-robin 分片调度（hybrid_dispatch.cuh:490-532） ===\n");
    struct Case { int R; int chunk; std::vector<int> tails; };
    std::vector<Case> cases = {
        {4, 6, {20, 13, 5, 0}},
        {8, 6, {1, 2, 3, 4, 5, 6, 7, 8}},
        {3, 100, {7, 7, 7}},           // chunk > tail：一步一片
        {5, 1, {9, 0, 0, 3, 0}},       // chunk=1：逐 slot 轮转
    };
    bool ok = true;
    for (auto& c : cases) {
        const int R = c.R, chunk = c.chunk;
        int *d_t, *d_seq, *d_len, *d_pc;
        CK(cudaMalloc(&d_t, R * sizeof(int)));
        CK(cudaMalloc(&d_seq, 4096 * sizeof(int)));
        CK(cudaMalloc(&d_len, sizeof(int)));
        CK(cudaMalloc(&d_pc, R * sizeof(int)));
        CK(cudaMemcpy(d_t, c.tails.data(), R * sizeof(int), cudaMemcpyHostToDevice));
        k_rr<<<1, 32>>>(d_t, R, chunk, d_seq, d_len, d_pc);
        CK(cudaDeviceSynchronize());
        int len = 0; std::vector<int> seq(4096), pc(R);
        CK(cudaMemcpy(&len, d_len, sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(seq.data(), d_seq, len * sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(pc.data(), d_pc, R * sizeof(int), cudaMemcpyDeviceToHost));

        // host 贪心 oracle（与设备同一规则）
        std::vector<int> ref_old(R, 0), ref_seq;
        int ref_recv = 0;
        while (true) {
            unsigned wip = 0;
            for (int r = 0; r < R; ++ r) if (c.tails[r] > ref_old[r]) wip |= (1u << r);
            if (!wip) break;
            const int offset = (ref_recv + 1) % R;
            unsigned hi = (wip >> offset) << offset;
            ref_recv = hi ? __builtin_ffs((int)hi) - 1 : __builtin_ffs((int)wip) - 1;
            const int start = ref_old[ref_recv];
            const int end = std::min(c.tails[ref_recv], start + chunk);
            ref_old[ref_recv] = end;
            ref_seq.push_back(ref_recv);
        }
        bool covered = (len == (int)ref_seq.size());
        for (int r = 0; r < R; ++ r) covered &= (pc[r] == c.tails[r]);
        bool same_seq = (len == (int)ref_seq.size());
        for (int i = 0; i < len and i < (int)ref_seq.size(); ++ i) same_seq &= (seq[i] == ref_seq[i]);
        ok &= covered and same_seq;
        printf("    R=%d chunk=%d tails={", R, chunk);
        for (int r = 0; r < R; ++ r) printf("%d%s", c.tails[r], r + 1 < R ? "," : "");
        printf("} steps=%d covered=%s seq_match=%s\n", len, covered ? "Y" : "N", same_seq ? "Y" : "N");
        cudaFree(d_t); cudaFree(d_seq); cudaFree(d_len); cudaFree(d_pc);
    }
    check(ok, "每 rank [0,tail) 恰好处理一次 + 选择序列 = 贪心 round-robin");
}

// -----------------------------------------------------------------------------
// Part 5: TokenLayout 字节几何（layout/ep/token.cuh:10-82）——hybrid 三层缓冲的每槽大小
//   隐藏字节数必须 32B(TMA) 对齐（:29 断言）；每槽 = align(h) + align(sf) + align(meta)
//   再 align 到 64B(RDMA)（:33-41）。
// -----------------------------------------------------------------------------
static void part5_token_layout() {
    printf("\n=== Part 5: TokenLayout 每槽字节（layout/ep/token.cuh:10-82） ===\n");
    struct Case { int hidden, sf, topk; bool meta; };
    std::vector<Case> cases = {
        {7168 * 1, 56, 8, true},   // bf16 hidden=7168, 56 个 sf pack
        {7168 * 2, 0, 8, false},   // combine：无 sf、无 metadata
    };
    bool ok = true;
    for (auto& c : cases) {
        const layout::TokenLayout tl(c.hidden, c.sf, c.topk, c.meta);
        // host oracle
        auto al = [](int a, int b) { return (a + b - 1) / b * b; };
        const int meta_bytes = c.topk * (4 + 4) + (c.meta ? (1 + c.topk) * 4 : 0);
        const int data = al(al(c.hidden, 32) + al(c.sf, 32) + al(meta_bytes, 32), 64);
        const int with_mb = data + al(32, 32);          // false 时 +0
        const bool good = (tl.get_num_bytes<false>() == data) and
                          (tl.get_num_bytes<true>() == with_mb);
        ok &= good;
        printf("    hidden=%-6d sf=%-3d topk=%d meta=%-5s bytes<false>=%d bytes<true>=%d %s\n",
               c.hidden, c.sf, c.topk, c.meta ? "true" : "false",
               tl.get_num_bytes<false>(), tl.get_num_bytes<true>(), good ? "" : "<<< BAD");
    }
    check(ok, "TokenLayout 三段对齐 + RDMA 64B 对齐 vs host oracle");
}

int main() {
    printf("DeepEP hybrid dispatch/combine —— 架构无关内核 H100 真机验证\n");
    printf("DeepEP commit 93eb6eb；真实 include：common/math.cuh、common/ptx.cuh、layout/ep/token.cuh\n");
    part1_sentinel();
    part2_pack();
    part3_psum();
    part4_rr();
    part5_token_layout();
    printf("\n================ 总计: PASS=%d FAIL=%d ================\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
