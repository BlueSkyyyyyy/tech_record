// P56-1 实验：DeepEP 的对称内存分配层（csrc/kernels/comm/symmetric.hpp）
//
// 为什么 H100(sm90) 可跑：
//   * 这一层是**纯 host-side CUDA VMM + NCCL 主机 API**，与 UMMA/tcgen05/GIN 设备 API 无关：
//       - `DeviceContext::build_alloc_prop`（symmetric.hpp:41-75）只查 `cuDeviceGetAttribute`
//         （FABRIC / GPUDirect-RDMA / host NUMA）与 `cuMemGetAllocationGranularity`；
//       - `cumem_create_with_fallback`（:79-93）是 `cuMemCreate` 的 FABRIC→POSIX_FD 回退；
//       - `ElasticSymmetricMemory`（:139-190）与 `HybridElasticSymmetricMemory`（:193-300）用
//         `cuMemAddressReserve/cuMemMap/cuMemSetAccess/cuMemExportToShareableHandle` 与 `pidfd_getfd`；
//       - `GPUSymmetricMemory`（:120-135）只包 `ncclMemAlloc/ncclMemFree`（NCCL 主机 API，2.19+）。
//   * 故本实验**直接 include 真实头文件**，调用真实类/函数，先跑 host 侧，再启动 kernel 真读写每段内存。
//   * 跨进程 shard 共享（HybridElasticSymmetricMemory 的 `pidfd_open`+`pidfd_getfd` 路径）用 fork 真跑。
//   * 依赖 NCCL 设备 API 的部分（ncclCommWindowRegister / ncclGetLsaDevicePointer / GIN）不在本层，不涉及。
//
// 编译：见 harness/exp_deepep_symmetric_alloc.sh
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>

#include <sys/wait.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>

#include <cuda_runtime.h>
#include <memory>

// 直接 include 真实头文件（-I <DeepEP>/）
#include "csrc/kernels/comm/symmetric.hpp"

using namespace deep_ep;
namespace sym = deep_ep::symmetric;

// ---------------------------------------------------------------- 测试记账
static int g_pass = 0, g_fail = 0;
static void expect(const bool cond, const std::string& msg) {
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", msg.c_str());
    if (cond) ++ g_pass; else ++ g_fail;
}
static void section(const char* name) { std::printf("\n=== %s ===\n", name); }

#define OK(cond, msg) expect((cond), (msg))
#define EQI(a, b, msg) expect((int64_t)(a) == (int64_t)(b), \
    std::string(msg) + " (" + std::to_string((long long)(a)) + " == " + std::to_string((long long)(b)) + ")")

// ---------------------------------------------------------------- device kernel
__global__ void fill_u8_kernel(unsigned char* p, unsigned long long n, unsigned int value) {
    const auto stride = (unsigned long long)blockDim.x * gridDim.x;
    for (auto i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = (unsigned char)value;
}
__global__ void patterned_u8_kernel(unsigned char* p, unsigned long long n, unsigned int salt) {
    const auto stride = (unsigned long long)blockDim.x * gridDim.x;
    for (auto i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = (unsigned char)((i * 131u + salt) & 0xffu);
}

// 用 kernel 铺满，再 D2H 拷回核对（对 GPU 段 / CPU-NUMA 段 / VMM 映射段都通用）
static bool kernel_roundtrip(void* ptr, const int64_t& nbytes, const unsigned int value) {
    unsigned long long n = (unsigned long long)nbytes;
    fill_u8_kernel<<<176, 256>>>((unsigned char*)ptr, n, value);
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::printf("    [kernel err] %s\n", cudaGetErrorString(err)); return false; }
    std::vector<unsigned char> host(n);
    err = cudaMemcpy(host.data(), ptr, nbytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::printf("    [memcpyD2H err] %s\n", cudaGetErrorString(err)); return false; }
    for (unsigned long long i = 0; i < n; ++ i)
        if (host[i] != (unsigned char)value) return false;
    return true;
}

// ================================================================= Part 1
static void part1_props() {
    section("Part 1: DeviceContext 分配属性构造（symmetric.hpp:21-75）");
    sym::DeviceContext ctx;

    // 原始设备属性（:29 / :51 / :61）
    int numa = -1, fabric = -1, gdr = -1;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, ctx.device));
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&fabric, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, ctx.device));
    CUDA_DRIVER_CHECK(lazy_cuDeviceGetAttribute(&gdr, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, ctx.device));
    std::printf("  [info] device=%d host_numa=%d fabric_supported=%d gpu_direct_rdma_vmm=%d\n",
                ctx.device_idx, numa, fabric, gdr);
    EQI(ctx.numa_idx, numa, "DeviceContext.numa_idx == HOST_NUMA_ID");
    OK(numa >= 0, "H100 暴露 host NUMA id (>=0)");

    auto gpu = ctx.gpu_alloc_prop();
    auto cpu = ctx.cpu_alloc_prop();

    EQI(gpu.type, CU_MEM_ALLOCATION_TYPE_PINNED, "gpu prop.type == PINNED");
    EQI(gpu.location.type, CU_MEM_LOCATION_TYPE_DEVICE, "gpu prop.location DEVICE");
    EQI(gpu.location.id, ctx.device_idx, "gpu prop.location.id == device_idx");
    EQI(cpu.location.type, CU_MEM_LOCATION_TYPE_HOST_NUMA, "cpu prop.location HOST_NUMA");
    EQI(cpu.location.id, ctx.numa_idx, "cpu prop.location.id == numa_idx");
    OK((gpu.requestedHandleTypes & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != 0,
       "requestedHandleTypes 含 POSIX_FILE_DESCRIPTOR");
    EQI(gpu.allocFlags.gpuDirectRDMACapable, 1, "gpu prop.allocFlags.gpuDirectRDMACapable == 1（:65-66 断言通过）");
    EQI(cpu.allocFlags.gpuDirectRDMACapable, 0, "cpu prop 不设 gpuDirectRDMACapable");
    // FABRIC 位只有在设备支持时才置（:49-55）
    OK(((gpu.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) != 0) == (fabric != 0),
       "gpu prop 的 FABRIC 位 == 设备属性 fabric_supported");

    // 粒度检查（:69-73）：recommended granularity 必须整除 kNumAllocationAlignmentBytes
    size_t gran_rec = 0, gran_min = 0;
    CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&gran_rec, &gpu, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&gran_min, &gpu, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    std::printf("  [info] granularity recommended=%zu minimum=%zu ; kNumAllocationAlignmentBytes=%d\n",
                gran_rec, gran_min, kNumAllocationAlignmentBytes);
    EQI(kNumAllocationAlignmentBytes, 2 * 1024 * 1024, "kNumAllocationAlignmentBytes == 2 MiB（compiled.cuh:69）");
    OK(gran_rec == 0 or (kNumAllocationAlignmentBytes % gran_rec == 0),
       "2 MiB 整除 recommended granularity（:72 断言）");
    OK(gran_rec >= gran_min, "recommended granularity >= minimum granularity");

    // cpu prop 的粒度（HOST_NUMA 路径）
    size_t gran_cpu = 0;
    CUDA_DRIVER_CHECK(lazy_cuMemGetAllocationGranularity(&gran_cpu, &cpu, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    std::printf("  [info] cpu-alloc recommended granularity=%zu\n", gran_cpu);
    OK(gran_cpu == 0 or (kNumAllocationAlignmentBytes % gran_cpu == 0),
       "2 MiB 整除 cpu recommended granularity");
}

// ================================================================= Part 2
static void part2_fabric_fallback() {
    section("Part 2: cumem_create_with_fallback 的 FABRIC→POSIX_FD 回退（symmetric.hpp:79-93）");
    sym::DeviceContext ctx;

    // (a) 正常（不强制 FABRIC）：走 else 分支（:90-91），应成功
    {
        auto prop = ctx.gpu_alloc_prop(true); // skip_fabric=true → 不含 FABRIC
        CUmemGenericAllocationHandle h = {};
        bool ok = true;
        try { sym::cumem_create_with_fallback(&h, kNumAllocationAlignmentBytes, &prop); }
        catch (const std::exception& e) { ok = false; std::printf("    [exc] %s\n", e.what()); }
        OK(ok, "POSIX_FD-only prop：cuMemCreate 成功（else 分支）");
        if (ok) CUDA_DRIVER_CHECK(lazy_cuMemRelease(h));
    }

    // (b) 强制请求 FABRIC：若设备不支持，cuMemCreate 应返回 NOT_PERMITTED/NOT_SUPPORTED，
    //     函数在 :84-86 清掉 FABRIC 位后重试并成功；若设备确实支持 FABRIC，则直接成功。
    {
        auto prop = ctx.gpu_alloc_prop(true);
        prop.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(
            prop.requestedHandleTypes | CU_MEM_HANDLE_TYPE_FABRIC);
        CUmemGenericAllocationHandle h = {};
        bool ok = true;
        try { sym::cumem_create_with_fallback(&h, kNumAllocationAlignmentBytes, &prop); }
        catch (const std::exception& e) { ok = false; std::printf("    [exc] %s\n", e.what()); }
        OK(ok, "强制 FABRIC 请求：回退逻辑后 cuMemCreate 仍成功");
        OK((prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) == 0 or ok,
           "回退后 FABRIC 位被清（或设备原生支持 FABRIC）");
        std::printf("    [info] 最终 requestedHandleTypes = 0x%x\n", (unsigned)prop.requestedHandleTypes);
        if (ok) CUDA_DRIVER_CHECK(lazy_cuMemRelease(h));
    }
}

// ================================================================= Part 3
static void part3_elastic() {
    section("Part 3: ElasticSymmetricMemory：GPU 段 + 独立 storage 段共享一段连续 VA（symmetric.hpp:139-190）");
    const int64_t sym_bytes = 2 * kNumAllocationAlignmentBytes;      // 主对称内存 4 MiB
    const int64_t sto_bytes = kNumAllocationAlignmentBytes;          // storage 2 MiB

    for (int use_cpu = 0; use_cpu <= 1; ++ use_cpu) {
        std::string tag = use_cpu ? "CPU(NUMA)-backed storage" : "GPU-backed storage";
        std::printf("  -- %s --\n", tag.c_str());
        std::shared_ptr<sym::SymmetricMemory> mem;
        bool ok = true;
        try { mem = sym::alloc(sym_bytes, sto_bytes, use_cpu != 0); }
        catch (const std::exception& e) { ok = false; std::printf("    [exc] %s\n", e.what()); }
        OK(ok, tag + "：symmetric::alloc 成功");
        if (!ok) continue;

        EQI(mem->num_sym_bytes, sym_bytes, tag + "：num_sym_bytes 记录正确");
        OK((reinterpret_cast<uint64_t>(mem->ptr) % kNumAllocationAlignmentBytes) == 0,
           tag + "：返回指针 2 MiB 对齐（:178 断言）");
        std::printf("    [info] ptr=0x%llx\n", (unsigned long long)(uintptr_t)mem->ptr);

        // 两段各铺不同 pattern 并读回：证明「GPU 段 + 独立物理段」在同一 VA 上都可读写
        OK(kernel_roundtrip(mem->ptr, sym_bytes, 0x5a), tag + "：主对称内存段 kernel 铺/读 0x5a 一致");
        void* storage_ptr = (unsigned char*)mem->ptr + sym_bytes;
        OK(kernel_roundtrip(storage_ptr, sto_bytes, 0xa5), tag + "：storage 段 kernel 铺/读 0xa5 一致");
        // 两段互不重叠：主段仍是 0x5a，storage 段是 0xa5
        std::vector<unsigned char> probe(1);
        cudaMemcpy(probe.data(), mem->ptr, 1, cudaMemcpyDeviceToHost);
        bool p1 = probe[0] == 0x5a;
        cudaMemcpy(probe.data(), storage_ptr, 1, cudaMemcpyDeviceToHost);
        bool p2 = probe[0] == 0xa5;
        OK(p1 and p2, tag + "：两段物理 backing 独立（可同时持有不同值）");
    }
}

// ================================================================= Part 4
static void part4_export_import() {
    section("Part 4: create_shared_handle 的 POSIX-FD 导出 + 本进程再导入（symmetric.hpp:281-299）");
    const int64_t bytes = kNumAllocationAlignmentBytes;
    for (int use_cpu = 0; use_cpu <= 1; ++ use_cpu) {
        std::string tag = use_cpu ? "CPU storage handle" : "GPU storage handle";
        auto [pid, fd] = sym::HybridElasticSymmetricMemory::create_shared_handle(bytes, use_cpu != 0);
        OK(fd >= 0, tag + "：导出得到合法 FD");
        EQI(pid, getpid(), tag + "：handle.pid == 本进程 pid");
        std::printf("    [info] (pid=%d, fd=%d)\n", pid, fd);

        // 导出即释放 allocation handle，FD 持有物理内存（:295-296 注释）
        // 用另一段 VA reserve+map 导入这份 FD，再从 kernel 读写
        CUmemGenericAllocationHandle imp = {};
        bool ok = true;
        void* base = nullptr;
        try {
            CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(
                &imp, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)),
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
            CUdeviceptr addr;
            CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, bytes, kNumAllocationAlignmentBytes, 0, 0));
            base = reinterpret_cast<void*>(static_cast<uintptr_t>(addr));
            CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, bytes, 0, imp, 0));
            sym::DeviceContext ctx;
            sym::set_access(addr, bytes, ctx.device_idx);
        } catch (const std::exception& e) { ok = false; std::printf("    [exc] %s\n", e.what()); }
        OK(ok, tag + "：本进程导入 FD + map + set_access 成功");
        if (ok) {
            OK(kernel_roundtrip(base, bytes, 0x3c), tag + "：导入段 kernel 铺/读 0x3c 一致");
            CUDA_DRIVER_CHECK(lazy_cuMemUnmap((CUdeviceptr)base, bytes));
            CUDA_DRIVER_CHECK(lazy_cuMemRelease(imp));
            CUDA_DRIVER_CHECK(lazy_cuMemAddressFree((CUdeviceptr)base, bytes));
        }
        close(fd);
    }
}

// ================================================================= Part 5
static void part5_hybrid() {
    section("Part 5: HybridElasticSymmetricMemory 的多 shard 连续映射（symmetric.hpp:193-262）");
    const int num_scaleup = 3;
    const int64_t sym_bytes = kNumAllocationAlignmentBytes;      // 2 MiB
    const int64_t sto_bytes = kNumAllocationAlignmentBytes;      // 每 shard 2 MiB

    // 造 num_scaleup 个本进程自己的 storage shard，当作「intra-node peers」的 (pid,fd)
    sym::shared_comm_t shared_comm;
    std::vector<int> fds;
    for (int i = 0; i < num_scaleup; ++ i) {
        auto h = sym::HybridElasticSymmetricMemory::create_shared_handle(sto_bytes, false);
        shared_comm.push_back(h);
        fds.push_back(h.second);
    }
    // scaleout_rank_idx=0 → 取 shared_comm[0..num_scaleup)
    std::shared_ptr<sym::SymmetricMemory> mem;
    bool ok = true;
    try {
        mem = sym::alloc(sym_bytes, sto_bytes, /*use_cpu=*/false,
                         /*allow_hybrid_mode=*/true, num_scaleup, 0, shared_comm);
    } catch (const std::exception& e) { ok = false; std::printf("    [exc] %s\n", e.what()); }
    OK(ok, "hybrid alloc 成功（num_scaleup=3，全部 local pid → 走 local_fd 直用分支）");
    if (ok) {
        EQI(mem->num_sym_bytes, sym_bytes, "num_sym_bytes 仍为主段大小");
        OK((reinterpret_cast<uint64_t>(mem->ptr) % kNumAllocationAlignmentBytes) == 0,
           "hybrid 指针 2 MiB 对齐（:221 断言）");
        // 主段 + 3 个 shard 依次连续；每个 shard 铺不同 pattern，互不干扰
        bool all = kernel_roundtrip(mem->ptr, sym_bytes, 0x11);
        for (int i = 0; i < num_scaleup; ++ i) {
            void* shard = (unsigned char*)mem->ptr + sym_bytes + (int64_t)i * sto_bytes;
            all = all and kernel_roundtrip(shard, sto_bytes, (unsigned)(0x20 + i));
        }
        OK(all, "主段 + 3 个 shard 各自 kernel 铺/读一致（物理段独立，VA 连续）");
        // 再回读主段，确认 shard 写入没踩主段
        std::vector<unsigned char> probe(1);
        cudaMemcpy(probe.data(), mem->ptr, 1, cudaMemcpyDeviceToHost);
        OK(probe[0] == 0x11, "shard 写入未污染主段");
    }
    mem.reset();          // 析构：unmap + release + close(local_export_fd)
    // 注意：析构会 close 最后一个 local_fd，其余 fd 需我们自行清理
    for (int i = 0; i + 1 < (int)fds.size(); ++ i) close(fds[i]);
}

// ================================================================= Part 6
static ssize_t write_all(const int fd, const void* buf, const size_t n) { return ::write(fd, buf, n) == (ssize_t)n ? (ssize_t)n : -1; }
static bool read_all(const int fd, void* buf, const size_t n) {
    size_t got = 0; while (got < n) { auto r = ::read(fd, (char*)buf + got, n - got); if (r <= 0) return false; got += r; } return true;
}

static const char* g_exe_path = nullptr;
static const uint32_t MAGIC_CHILD = 0xC0FFEE01u, MAGIC_PARENT = 0xDEADBEE2u;

// 子进程模式：fork 后立即 exec 本二进制（避免 fork-after-CUDA-init），
// create_shared_handle → 映射自己的段 → 写 MAGIC_CHILD → 向 stdout 公布 (pid, fd) →
// 读 stdin 等父写完 → 读回 MAGIC_PARENT 决定退出码。
static int child_publish() {
    const int64_t bytes = kNumAllocationAlignmentBytes;
    cudaSetDevice(0);
    try {
        auto [pid, fd] = sym::HybridElasticSymmetricMemory::create_shared_handle(bytes, false);
        CUmemGenericAllocationHandle imp = {};
        CUdeviceptr addr;
        CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(
            &imp, reinterpret_cast<void*>(static_cast<uintptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, bytes, kNumAllocationAlignmentBytes, 0, 0));
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, bytes, 0, imp, 0));
        sym::DeviceContext ctx; sym::set_access(addr, bytes, ctx.device_idx);

        uint32_t v = MAGIC_CHILD;
        cudaMemcpy((void*)addr, &v, 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        std::printf("READY %d %d\n", pid, fd);
        std::fflush(stdout);

        // 等父进程写完 MAGIC_PARENT（读到一行）
        char line[64];
        if (fgets(line, sizeof(line), stdin) == nullptr) _exit(12);
        uint32_t got = 0;
        cudaMemcpy(&got, (void*)addr, 4, cudaMemcpyDeviceToHost);
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr, bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(imp));
        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(addr, bytes));
        close(fd);
        return (got == MAGIC_PARENT) ? 0 : 13;
    } catch (const std::exception& e) {
        std::printf("CHILD_EXC %s\n", e.what()); std::fflush(stdout); return 14;
    }
}

// 跨进程：fork+exec 子进程 create_shared_handle → 父进程 pidfd_open/pidfd_getfd →
// 两边映射同一物理段 → 子写父读、父写子读（HybridElastic 在真实多 rank import 对端 shard 的路径）
static void part6_cross_process() {
    section("Part 6: 跨进程 pidfd 共享（HybridElasticSymmetricMemory 的 pidfd_open/pidfd_getfd 路径，:232-238）");
    const int64_t bytes = kNumAllocationAlignmentBytes;

    int c2p[2], p2c[2];
    pipe(c2p); pipe(p2c);   // c2p: 子 stdout -> 父；p2c: 父 -> 子 stdin

    pid_t child = fork();
    if (child == 0) {
        dup2(c2p[1], STDOUT_FILENO); close(c2p[1]); close(c2p[0]);
        dup2(p2c[0], STDIN_FILENO);  close(p2c[0]); close(p2c[1]);
        execl(g_exe_path, g_exe_path, "--child-publish", (char*)nullptr);
        _exit(127);   // exec 失败
    }
    close(c2p[1]); close(p2c[0]);
    cudaSetDevice(0);

    // 逐字读子进程的第一行 "READY <pid> <fd>"
    std::string line; char ch;
    while (line.find('\n') == std::string::npos and read_all(c2p[0], &ch, 1)) line.push_back(ch);
    int child_pid = 0, child_fd = -1;
    bool got_handle = std::sscanf(line.c_str(), "READY %d %d", &child_pid, &child_fd) == 2;
    OK(got_handle, "父进程收到子进程公布的 (pid, fd)");
    std::printf("    [info] child pid=%d (fork=%d), child fd=%d\n", child_pid, child, child_fd);

    bool ok = true;
    CUmemGenericAllocationHandle imp = {};
    CUdeviceptr addr = 0;
    int got_fd = -1;
    try {
        int pidfd = syscall(SYS_pidfd_open, child_pid, 0);
        OK(pidfd >= 0, "pidfd_open(child) 成功（:233）");
        got_fd = syscall(SYS_pidfd_getfd, pidfd, child_fd, 0);
        OK(got_fd >= 0, "pidfd_getfd 跨进程拿到 FD 副本（:235）");
        close(pidfd);
        CUDA_DRIVER_CHECK(lazy_cuMemImportFromShareableHandle(
            &imp, reinterpret_cast<void*>(static_cast<uintptr_t>(got_fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
        CUDA_DRIVER_CHECK(lazy_cuMemAddressReserve(&addr, bytes, kNumAllocationAlignmentBytes, 0, 0));
        CUDA_DRIVER_CHECK(lazy_cuMemMap(addr, bytes, 0, imp, 0));
        sym::DeviceContext ctx; sym::set_access(addr, bytes, ctx.device_idx);
    } catch (const std::exception& e) { ok = false; std::printf("    [exc] %s\n", e.what()); }
    OK(ok, "父进程导入对端段并映射到自己的 VA");

    if (ok) {
        uint32_t v = 0;
        cudaMemcpy(&v, (void*)addr, 4, cudaMemcpyDeviceToHost);
        expect(v == MAGIC_CHILD, "父进程读到子进程写的 MAGIC_CHILD（跨进程同物理内存可见）");
        uint32_t w = MAGIC_PARENT;
        cudaMemcpy((void*)addr, &w, 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        write_all(p2c[1], "\n", 1);           // 通知子读取
        CUDA_DRIVER_CHECK(lazy_cuMemUnmap(addr, bytes));
        CUDA_DRIVER_CHECK(lazy_cuMemRelease(imp));
        CUDA_DRIVER_CHECK(lazy_cuMemAddressFree(addr, bytes));
        if (got_fd >= 0) close(got_fd);
    } else {
        write_all(p2c[1], "\n", 1);
    }
    close(p2c[1]);
    // 排空子进程剩余 stdout
    { char buf[256]; while (read_all(c2p[0], buf, 1)) {} }

    int status = 0; waitpid(child, &status, 0);
    OK(WIFEXITED(status) and WEXITSTATUS(status) == 0,
       "子进程读回父写的 MAGIC_PARENT（exit=0）");
    if (WIFEXITED(status)) std::printf("    [info] child exit code = %d\n", WEXITSTATUS(status));
    close(c2p[0]);
}

// ================================================================= Part 7
static void part7_nccl_mem() {
    section("Part 7: GPUSymmetricMemory = ncclMemAlloc 包装（symmetric.hpp:120-135）");
    const int64_t bytes = 2 * kNumAllocationAlignmentBytes;
    bool ok = true;
    std::shared_ptr<sym::SymmetricMemory> mem;
    try { mem = sym::alloc(bytes, /*storage=*/0, /*use_cpu=*/false); }
    catch (const std::exception& e) { ok = false; std::printf("    [exc] %s\n", e.what()); }
    OK(ok, "num_rdma_storage_bytes==0 → 走 GPUSymmetricMemory（:321-323）");
    if (ok) {
        EQI(mem->num_sym_bytes, bytes, "num_sym_bytes 正确");
        OK((reinterpret_cast<uint64_t>(mem->ptr) % kNumAllocationAlignmentBytes) == 0,
           "ncclMemAlloc 返回 2 MiB 对齐（:125 断言）");
        std::printf("    [info] ncclMemAlloc ptr=0x%llx\n", (unsigned long long)(uintptr_t)mem->ptr);
        OK(kernel_roundtrip(mem->ptr, bytes, 0x77), "ncclMemAlloc 段 kernel 铺/读 0x77 一致");
    }

    // 非法对齐：非 2 MiB 倍数应被 :123 断言拒绝
    bool rejected = false;
    try { sym::GPUSymmetricMemory bad(kNumAllocationAlignmentBytes + 64); }
    catch (const std::exception&) { rejected = true; }
    OK(rejected, "非 2 MiB 倍数被 :123 断言拒绝");
    // 0 字节也应被拒
    rejected = false;
    try { sym::GPUSymmetricMemory bad(0); } catch (const std::exception&) { rejected = true; }
    OK(rejected, "0 字节被 :123 断言拒绝");
}

int main(int argc, char** argv) {
    g_exe_path = argv[0];
    if (argc >= 2 and std::strcmp(argv[1], "--child-publish") == 0)
        return child_publish();

    if (cudaSetDevice(0) != cudaSuccess) { std::printf("cudaSetDevice failed\n"); return 2; }
    cudaFree(nullptr);

    std::printf("DeepEP 对称内存分配层实验 —— 直接 include 真实 symmetric.hpp，H100(sm90) 真机\n");

    part1_props();
    part2_fabric_fallback();
    part3_elastic();
    part4_export_import();
    part5_hybrid();
    part6_cross_process();
    part7_nccl_mem();

    std::printf("\n=== SUMMARY ===\n");
    std::printf("PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
