// P24-1 实验：SM 局部域（locality domain）探针与内存域分配。
//
// 为什么能在 H100(sm90) 上跑：`deep_gemm/include/deep_gemm/impls/sm100_locality_domain.cuh`
// 里的 `locality_domain_probe_chase` 只用 `%smid` + `ld.global.cg` + `clock64`，与
// UMMA/tcgen05/MXFP 都无关（文件名叫 sm100 只是因为 DeepGEMM 2.x 只面向 SM100 发布）；
// host 侧 `csrc/apis/locality_domain.hpp` 的 `probe / try_probe / balance / get_even`
// 全是纯张量逻辑，架构无关。**MLOPart 本地化内存在本机不可用**（需要 MPS `-mlopart` +
// 双 device instance），所以本实验：
//   (A) 原样编译运行真实 device 探针，量 device 内存的 per-hop 延迟（两个「域」都放 HBM）；
//   (B) 用一个「远端域」模拟本地化差异（domain 1 放 host-pinned mapped memory），
//       验证探针**能检测出 far/near 差异并给出稳定、TPC 一致的域划分**；
//   (C) 用 C++ 独立复刻 host 侧 `try_probe / get_even / balance` 逻辑，跑边界用例；
//   (D) 对照 DeepGEMM 真机 API：`is_localization_available()==False` 时
//       `get_sm_locality_domains()` 是否退化为 even 映射（见 harness 里的 python 段）。
//
// 编译：见 harness/exp_locality_domain_probe.sh

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>

#include <cuda_runtime.h>

// 真实设备代码（原样 include，DTO 由 DeepGEMM 提供；仅用 smid / ld.global.cg / clock64）
#include <deep_gemm/impls/sm100_locality_domain.cuh>

using namespace deep_gemm;

// ---------------------------------------------------------------------------
// 与 csrc/apis/locality_domain.hpp 一致的可调常量（行号见 docs/34）
// ---------------------------------------------------------------------------
static constexpr int kNumLocalityDomainProbeChunkBytes = 4096;
static constexpr int kNumLocalityDomainProbeLineBytes  = 128;
static constexpr int kNumLocalityDomainProbeHops       = kNumLocalityDomainProbeChunkBytes / kNumLocalityDomainProbeLineBytes;  // 32
static constexpr int kNumSMsPerTPC                     = 2;
static constexpr int kNumLinesPerChunk                 = kNumLocalityDomainProbeHops;   // 32
static constexpr int kNumWordsPerLine                  = kNumLocalityDomainProbeLineBytes / (int)sizeof(int);  // 32
static constexpr int kNumProbeChunksPerSM              = 8;    // locality_domain.hpp:63
static constexpr int kChainLineStride                  = 7;    // locality_domain.hpp:65
static constexpr float kMinFarNearRatio                = 1.25f;  // locality_domain.hpp:66
static constexpr int kNumMaxProbeAttempts              = 5;    // locality_domain.hpp:67 (host: probe 最多重试 5 次)
static constexpr int kNumDomains                       = 2;    // types.cuh:54 kNumDeviceLocalityDomains

#define CUDA_CHECK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while (0)

// 一个 chunk（4KB = 32 行 128B）的链：每行的 word0 指向第 (l + 7) % 32 行。
// 7 与 32 互质 => 单条链恰好访问全部 32 行、每行一次（locality_domain.hpp:73-79）。
static void build_chain(uint32_t* chunk) {
    std::fill(chunk, chunk + kNumLinesPerChunk * kNumWordsPerLine, 0u);
    for (int l = 0; l < kNumLinesPerChunk; ++l)
        chunk[l * kNumWordsPerLine + 0] = (uint32_t)(((l + kChainLineStride) % kNumLinesPerChunk) * kNumWordsPerLine);
}

// ---------------------------------------------------------------------------
// host 侧参考实现（复刻 csrc/apis/locality_domain.hpp:82-142）
// ---------------------------------------------------------------------------
// torch.median 对偶数个元素取「下中位数」=> 索引 (n-1)/2
static uint16_t lower_median(std::vector<uint16_t> v) {
    std::sort(v.begin(), v.end());
    return v[(v.size() - 1) / 2];
}

struct ProbeResult {
    std::vector<uint8_t> sm_domain;   // 每个 SM 的域
    float ratio_min;                  // 对所有 SM 的最坏 far/near 比
    bool clear;                       // latency[1] >= latency[0]*1.25 全成立
    bool consistent;                  // 一个 TPC 内两个 SM 同域
    bool valid;                       // clear && consistent
    std::vector<float> near_lat, far_lat;
};

// med[domain][sm] 是「已经对 chunk 取过中位数」的逐 SM 延迟
static ProbeResult try_probe_rows(const std::vector<std::vector<float>>& med) {
    const int N = (int)med[0].size();
    ProbeResult r;
    r.sm_domain.resize(N);
    r.near_lat.resize(N); r.far_lat.resize(N);
    r.ratio_min = 1e30f;
    for (int sm = 0; sm < N; ++sm) {
        float a = med[0][sm], b = med[1][sm];
        const bool d = (a <= b) ? 0 : 1;           // order[0]: 较小者为 near（并列取 0）
        const float near = d ? b : a, far = d ? a : b;
        r.sm_domain[sm] = (uint8_t)d;
        r.near_lat[sm] = near; r.far_lat[sm] = far;
        r.ratio_min = std::min(r.ratio_min, far / near);
    }
    r.clear = true;
    for (int sm = 0; sm < N; ++sm) r.clear = r.clear && (r.far_lat[sm] >= r.near_lat[sm] * kMinFarNearRatio);
    r.consistent = true;
    for (int t = 0; t + 1 < N; t += kNumSMsPerTPC)
        r.consistent = r.consistent && (r.sm_domain[t] == r.sm_domain[t + 1]);
    r.valid = r.clear && r.consistent;
    return r;
}

static std::vector<uint8_t> get_even(int num_sms) {
    std::vector<uint8_t> out(num_sms);
    for (int i = 0; i < num_sms; ++i) out[i] = (uint8_t)((i / kNumSMsPerTPC) % kNumDomains);
    return out;
}

static std::vector<uint8_t> balance(std::vector<uint8_t> sm_domain) {
    const int N = (int)sm_domain.size();
    const int num_tpcs = N / kNumSMsPerTPC;
    int num_domain_1 = 0;
    for (int t = 0; t < num_tpcs; ++t) num_domain_1 += sm_domain[t * kNumSMsPerTPC];
    const uint8_t larger = (num_domain_1 * 2 > num_tpcs) ? 1 : 0;
    int need = num_tpcs / 2, moved = 0;
    for (int t = 0; t < num_tpcs && moved < need; ++t) {
        if (sm_domain[t * kNumSMsPerTPC] == larger) {
            sm_domain[t * kNumSMsPerTPC] = sm_domain[t * kNumSMsPerTPC + 1] = (uint8_t)(larger ^ 1);
            ++moved;
        }
    }
    return sm_domain;
}

// ---------------------------------------------------------------------------
// device 侧：跑真实探针 kernel，返回逐 (domain, sm, chunk) 的 cycles/hop
// ---------------------------------------------------------------------------
static void run_probe(uint32_t* const buf[kNumDomains], std::vector<std::vector<uint16_t>>& latency_flat, int num_sms) {
    // latency[domain][sm*chunks + chunk]
    latency_flat.assign(kNumDomains, std::vector<uint16_t>((size_t)num_sms * kNumProbeChunksPerSM, 0));
    auto launch = [&](int d) {
        std::vector<uint16_t> lat((size_t)num_sms * kNumProbeChunksPerSM);
        uint16_t* d_lat = nullptr;
        CUDA_CHECK(cudaMalloc(&d_lat, lat.size() * sizeof(uint16_t)));
        // 真实 device kernel：grid = num_sms，block = 32（一个 warp）
        locality_domain_probe_chase<kNumLocalityDomainProbeHops, kNumProbeChunksPerSM, kNumLocalityDomainProbeChunkBytes>
            <<<num_sms, 32>>>(buf[d], d_lat);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(lat.data(), d_lat, lat.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        latency_flat[d] = lat;
        cudaFree(d_lat);
    };
    launch(0);
    launch(1);
}

static std::vector<std::vector<float>> median_per_sm(const std::vector<std::vector<uint16_t>>& lat, int num_sms) {
    std::vector<std::vector<float>> med(kNumDomains, std::vector<float>(num_sms));
    for (int d = 0; d < kNumDomains; ++d)
        for (int sm = 0; sm < num_sms; ++sm) {
            std::vector<uint16_t> v(kNumProbeChunksPerSM);
            for (int c = 0; c < kNumProbeChunksPerSM; ++c)
                v[c] = lat[d][(size_t)sm * kNumProbeChunksPerSM + c];
            med[d][sm] = lower_median(v);
        }
    return med;
}

static void print_hist(const char* tag, const std::vector<float>& v) {
    auto s = v;
    std::sort(s.begin(), s.end());
    printf("    %-13s min=%7.1f  med=%7.1f  max=%7.1f\n", tag, s.front(), s[(s.size() - 1) / 2], s.back());
}

int main() {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int num_sms = prop.multiProcessorCount;
    (void)kNumMaxProbeAttempts;  // 源码 probe_sm_locality_domains 最多重试 5 次（本实验单次）
    printf("=== P24-1 SM locality domain probe (real device kernel, H100) ===\n");
    printf("device=%s  sm_%d%d  num_sms=%d  L2=%d MB\n",
           prop.name, prop.major, prop.minor, num_sms, (int)(prop.l2CacheSize >> 20));

    // 校验链确实是「访问全部 32 行一次」的置换
    {
        std::vector<uint32_t> chunk(kNumLinesPerChunk * kNumWordsPerLine);
        build_chain(chunk.data());
        std::vector<int> visits(kNumLinesPerChunk, 0);
        uint32_t idx = chunk[0];
        for (int h = 0; h < kNumLinesPerChunk; ++h) { visits[idx / kNumWordsPerLine]++; idx = chunk[idx]; }
        const bool perm = (std::count(visits.begin(), visits.end(), 1) == kNumLinesPerChunk) && idx == chunk[0];
        printf("\n[A] probe chain: stride=%d lines/chunk=%d -> visits all lines once = %s\n",
               kChainLineStride, kNumLinesPerChunk, perm ? "YES" : "NO");
    }

    const size_t chunk_bytes = kNumLocalityDomainProbeChunkBytes;
    const size_t domain_bytes = (size_t)num_sms * kNumProbeChunksPerSM * chunk_bytes;

    // 256MB 刷 L2（locality_domain.hpp:69-71）
    void* flush_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_buf, 256ull << 20));

    // ---------------- (A) 两个域都在 device HBM ----------------
    printf("\n[B] two DEVICE-memory domains (no MLOPart -> expect no stable mapping)\n");
    {
        uint32_t* buf[kNumDomains];
        for (int d = 0; d < kNumDomains; ++d) {
            CUDA_CHECK(cudaMalloc(&buf[d], domain_bytes));
            CUDA_CHECK(cudaMemset(buf[d], 0, domain_bytes));
        }
        std::vector<uint32_t> chain(kNumLinesPerChunk * kNumWordsPerLine);
        build_chain(chain.data());
        // get_probe_chain 把同一个链铺到每个 chunk（buf.view(...).copy_(chain)）
        std::vector<uint32_t> host_chunks(domain_bytes / 4);
        for (size_t c = 0; c < domain_bytes / chunk_bytes; ++c)
            std::memcpy(host_chunks.data() + c * (chunk_bytes / 4), chain.data(), chain.size() * 4);
        for (int d = 0; d < kNumDomains; ++d) CUDA_CHECK(cudaMemcpy(buf[d], host_chunks.data(), domain_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(flush_buf, 0, 256ull << 20));  // flush_l2()

        std::vector<std::vector<uint16_t>> lat;
        run_probe(buf, lat, num_sms);
        auto med = median_per_sm(lat, num_sms);
        auto r = try_probe_rows(med);
        print_hist("domain0", med[0]);
        print_hist("domain1", med[1]);
        printf("    try_probe: clear=%s consistent=%s valid=%s  worst ratio=%.3f (need >=%.2f)\n",
               r.clear ? "T" : "F", r.consistent ? "T" : "F", r.valid ? "T" : "F", r.ratio_min, kMinFarNearRatio);
        auto even = get_even(num_sms);
        int c0 = (int)std::count(even.begin(), even.end(), 0), c1 = (int)std::count(even.begin(), even.end(), 1);
        printf("    -> fallback get_even_sm_locality_domains: %d/%d SMs per domain, TPC-consistent=%s\n",
               c0, c1, (r.sm_domain.size() ? "by construction" : ""));
        for (int d = 0; d < kNumDomains; ++d) cudaFree(buf[d]);
    }

    // ---------------- (B) 「远端域」= host-pinned mapped memory ----------------
    printf("\n[C] domain0=HBM, domain1=HOST-PINNED (simulated far domain -> expect detection)\n");
    {
        uint32_t* buf[kNumDomains];
        CUDA_CHECK(cudaMalloc(&buf[0], domain_bytes));
        CUDA_CHECK(cudaMemset(buf[0], 0, domain_bytes));

        uint32_t* host_ptr = nullptr;
        CUDA_CHECK(cudaHostAlloc(&host_ptr, domain_bytes, cudaHostAllocMapped));
        uint32_t* dev_ptr = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0));
        buf[1] = dev_ptr;

        std::vector<uint32_t> chain(kNumLinesPerChunk * kNumWordsPerLine);
        build_chain(chain.data());
        std::vector<uint32_t> host_chunks(domain_bytes / 4);
        for (size_t c = 0; c < domain_bytes / chunk_bytes; ++c)
            std::memcpy(host_chunks.data() + c * (chunk_bytes / 4), chain.data(), chain.size() * 4);
        CUDA_CHECK(cudaMemcpy(buf[0], host_chunks.data(), domain_bytes, cudaMemcpyHostToDevice));
        std::memcpy(host_ptr, host_chunks.data(), domain_bytes);
        CUDA_CHECK(cudaMemset(flush_buf, 0, 256ull << 20));

        std::vector<std::vector<uint16_t>> lat;
        run_probe(buf, lat, num_sms);
        auto med = median_per_sm(lat, num_sms);
        auto r = try_probe_rows(med);
        print_hist("domain0(HBM)", med[0]);
        print_hist("domain1(HOST)", med[1]);
        printf("    try_probe: clear=%s consistent=%s valid=%s  worst ratio=%.3f\n",
               r.clear ? "T" : "F", r.consistent ? "T" : "F", r.valid ? "T" : "F", r.ratio_min);
        int c1 = (int)std::count(r.sm_domain.begin(), r.sm_domain.end(), 1);
        printf("    derived sm_domain: domain0=%d domain1=%d SMs (should be all 0: HBM is near)\n",
               num_sms - c1, c1);
        auto bal = balance(r.sm_domain);
        int b0 = (int)std::count(bal.begin(), bal.end(), 0), b1 = (int)std::count(bal.begin(), bal.end(), 1);
        int moved = 0;
        for (int i = 0; i < num_sms; ++i) moved += (bal[i] != r.sm_domain[i]);
        printf("    balance: domain0=%d domain1=%d SMs, moved %d SMs (~minimal to even split)\n",
               b0, b1, moved);

        cudaFree(buf[0]);
        CUDA_CHECK(cudaFreeHost(host_ptr));
    }

    // ---------------- (C) host 侧逻辑边界用例 ----------------
    printf("\n[D] host-side try_probe / get_even / balance edge cases\n");
    {
        auto mk = [](std::vector<float> a, std::vector<float> b) {
            return std::vector<std::vector<float>>{a, b}; };
        int pass = 0, fail = 0;
        auto check = [&](const char* name, bool ok) {
            printf("    %-46s %s\n", name, ok ? "PASS" : "FAIL"); ok ? ++pass : ++fail; };

        // 1) 清晰两域（TPC 成对一致：偶 TPC near=0，奇 TPC near=1）
        {
            std::vector<float> a, b;
            for (int t = 0; t < 66; ++t) { float n = 100, f = 200; if (t % 2) std::swap(n, f); a.push_back(n); a.push_back(n); b.push_back(f); b.push_back(f); }
            auto r = try_probe_rows(mk(a, b));
            check("clear & TPC-consistent mapping accepted", r.valid && r.ratio_min >= 1.25f);
            int c1 = (int)std::count(r.sm_domain.begin(), r.sm_domain.end(), 1);
            check("  -> 66/66 SMs per domain", c1 == 66);
        }
        // 2) far/near 比不足（fuzzy）=> 拒绝
        {
            std::vector<float> a(132, 100.0f), b(132, 110.0f);  // ratio 1.10 < 1.25
            auto r = try_probe_rows(mk(a, b));
            check("ratio 1.10 < 1.25 rejected", !r.clear && !r.valid);
        }
        // 3) 比例刚好到阈值 => 接受
        {
            std::vector<float> a(132, 100.0f), b(132, 125.0f);
            auto r = try_probe_rows(mk(a, b));
            check("ratio exactly 1.25 accepted", r.clear);
        }
        // 4) TPC 内两 SM 不一致 => 拒绝
        {
            std::vector<float> a, b;
            for (int i = 0; i < 132; ++i) { bool near0 = (i % 2 == 0); a.push_back(near0 ? 100 : 200); b.push_back(near0 ? 200 : 100); }
            auto r = try_probe_rows(mk(a, b));
            check("TPC-inconsistent (SM0!=SM1) rejected", !r.consistent && !r.valid);
        }
        // 5) balance：源码 (locality_domain.hpp:126-136) 只有在输入「一个域占满」时才给出精确 50/50
        //    （这正是探针在真实 MLOPart 下的输出形态：进程的 SM 全部靠近同一物理域）。
        {
            std::vector<uint8_t> d0(132, 0), d1(132, 1);
            auto b0 = balance(d0), b1 = balance(d1);
            int a0 = (int)std::count(b0.begin(), b0.end(), 0), a1 = (int)std::count(b1.begin(), b1.end(), 1);
            check("balance(all-0) -> 66/66 SMs", a0 == 66);
            check("balance(all-1) -> 66/66 SMs", a1 == 66);
            // 非「占满」输入：源码翻转 num_tpcs/2 个 larger 域 TPC，不做一般化最小移动（记录该性质）
            std::vector<uint8_t> d(132, 0);
            for (int t = 0; t < 50; ++t) d[t * 2] = d[t * 2 + 1] = 1;   // 50/66 TPC of domain 1
            auto bal = balance(d);
            int c1 = (int)std::count(bal.begin(), bal.end(), 1);
            check("balance(50/66) NOT a general balancer (17/49) [documented]", c1 == 34);
        }
        // 6) get_even 对 8 SM：0,0,1,1,0,0,1,1
        {
            auto e = get_even(8);
            std::vector<uint8_t> want{0, 0, 1, 1, 0, 0, 1, 1};
            check("get_even(8) == [0,0,1,1,0,0,1,1]", e == want);
        }
        printf("    edge cases: %d PASS / %d FAIL\n", pass, fail);
    }

    cudaFree(flush_buf);
    printf("\ndone.\n");
    return 0;
}
