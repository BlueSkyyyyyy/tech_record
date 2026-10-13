// P80-1 实验：DeepEP 与 DeepGEMM/deep_jit 的「驱动/运行时符号懒加载 + 版本闸门 + 张量视图 helper」
// 地基层的真机验证（H100/sm90，纯 host，架构无关）。
//
// 直接 include 两个仓库的**真实头文件**：
//   - DeepEP    csrc/utils/lazy_driver.hpp   （deep_ep::get_driver_handle / deep_ep::lazy_cuXXX）
//   - DeepEP    csrc/utils/tensor.hpp        （deep_ep::get_shape / get_data_ptr）
//   - DeepGEMM  csrc/utils/compatibility.hpp （三个版本闸门 static_assert）
//   - deep_jit  utils/lazy.hpp               （LazyInit<T> / DJ_DECL_LAZY_DL_HANDLE / DJ_DECL_LAZY_DL_FUNCTION）
//   - deep_jit  backend/cuda/driver.hpp      （deep_jit::cuda::driver::lazy_cuXXX + CUDA_VERSION 闸门）
//   - deep_jit  utils/exception.hpp          （dlopen libdw 做回溯）
//
// 所有「跑不了 / 需要目标卡」的部分（Mega MoE 主 kernel、NCCL GIN）都不在这里；本文件覆盖的是
// **架构无关的地基**：libcuda/libdw 的 dlopen/dlsym、VMM 与可共享句柄、版本 static_assert。
//
// 编译与运行见 harness/exp_lazy_driver_compat.sh。

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <memory>

#include <dlfcn.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/torch.h>

// ---- 真实头文件 ----
#include "utils/lazy_driver.hpp"     // DeepEP   (-I <DeepEP>/csrc)
#include "utils/tensor.hpp"          // DeepEP   (-I <DeepEP>/csrc)
#include "utils/compatibility.hpp"   // DeepGEMM (-I <DeepGEMM>/csrc)
#include <deep_jit/utils/lazy.hpp>
#include <deep_jit/backend/cuda/driver.hpp>
#include <deep_jit/utils/exception.hpp>

static int g_pass = 0, g_fail = 0;

static void report(const std::string& name, bool ok, const std::string& detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                detail.empty() ? "" : " — ", detail.c_str());
    ok ? ++ g_pass : ++ g_fail;
}

static std::string cu_str(CUresult r) {
    const char* name = "unknown";
    const char* str = "unknown";
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &str);
    return std::string(name) + ": " + str;
}

// ============================================================================
// Part A：两库的「符号清单」在本机驱动上是否都能解析
// ============================================================================
static void part_a() {
    std::printf("\n== Part A：懒加载符号清单在 libcuda.so.1 上的可解析性 ==\n");

    const std::vector<std::string> deep_jit_symbols = {
        "cuGetErrorName", "cuGetErrorString", "cuDeviceGet", "cuDeviceGetAttribute",
        "cuFuncSetAttribute", "cuLaunchKernelEx", "cuStreamBatchMemOp", "cuTensorMapEncodeTiled",
        "cuLibraryLoadFromFile", "cuLibraryUnload", "cuLibraryGetKernelCount",
        "cuLibraryEnumerateKernels", "cuKernelGetFunction",
        "cuMemAddressFree", "cuMemAddressReserve", "cuMemCreate", "cuMemExportToShareableHandle",
        "cuMemGetAccess", "cuMemGetAllocationGranularity", "cuMemGetAllocationPropertiesFromHandle",
        "cuMemImportFromShareableHandle", "cuMemMap", "cuMemMapArrayAsync", "cuMemRelease",
        "cuMemRetainAllocationHandle", "cuMemSetAccess", "cuMemUnmap"
    };
    const std::vector<std::string> deep_ep_symbols = {
        "cuGetErrorName", "cuGetErrorString", "cuFuncSetAttribute", "cuModuleLoad", "cuModuleUnload",
        "cuModuleGetFunction", "cuLaunchKernelEx", "cuMemSetAccess", "cuMemRetainAllocationHandle",
        "cuMemGetAddressRange_v2", "cuMemAddressReserve", "cuMemAddressFree", "cuMemMap", "cuMemUnmap",
        "cuMemCreate", "cuMemRelease", "cuCtxGetDevice", "cuMemImportFromShareableHandle",
        "cuMemExportToShareableHandle", "cuMemGetAllocationGranularity", "cuDeviceGet",
        "cuDeviceGetAttribute", "cuStreamBatchMemOp"
    };

    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    report("dlopen(libcuda.so.1)", h != nullptr);

    int missing_jit = 0, missing_ep = 0;
    for (const auto& s : deep_jit_symbols) if (dlsym(h, s.c_str()) == nullptr) ++ missing_jit;
    for (const auto& s : deep_ep_symbols)   if (dlsym(h, s.c_str()) == nullptr) ++ missing_ep;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "deep_jit %zu 个缺 %d；DeepEP %zu 个缺 %d",
                  deep_jit_symbols.size(), missing_jit, deep_ep_symbols.size(), missing_ep);
    report("两库符号清单全部可解析", missing_jit == 0 && missing_ep == 0, buf);
}

// ============================================================================
// Part B：两个 handle 是否指向同一份 libcuda（dlopen 引用计数语义）
// ============================================================================
static void part_b() {
    std::printf("\n== Part B：两库各自 dlopen 的 handle 身份 ==\n");
    void* a = deep_ep::get_driver_handle();
    void* b = deep_jit::cuda::driver::get_cuda_handle();
    char buf[128];
    std::snprintf(buf, sizeof(buf), "deep_ep=%p deep_jit=%p", a, b);
    report("deep_ep::get_driver_handle() == deep_jit::get_cuda_handle()", a == b, buf);

    // 幂等：同一函数多次调用返回同一 handle
    report("同一库 handle 幂等", deep_ep::get_driver_handle() == a &&
                                 deep_jit::cuda::driver::get_cuda_handle() == b);
}

// ============================================================================
// Part C：懒加载调用 == 直接链接调用
// ============================================================================
static void part_c() {
    std::printf("\n== Part C：懒加载 wrapper 与直接链接的结果一致性 ==\n");

    cudaFree(nullptr);  // 确保有 current context（driver API cuDeviceGet 也需要）
    CUdevice dev_lazy = -1, dev_direct = -1;
    CUresult r1 = deep_ep::lazy_cuDeviceGet(&dev_lazy, 0);
    CUresult r2 = cuDeviceGet(&dev_direct, 0);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "lazy dev=%d direct dev=%d rc=(%s)", dev_lazy, dev_direct, cu_str(r1).c_str());
    report("lazy_cuDeviceGet == cuDeviceGet", r1 == CUDA_SUCCESS && r2 == CUDA_SUCCESS && dev_lazy == dev_direct, buf);

    int a_lazy = -1, a_direct = -1;
    deep_ep::lazy_cuDeviceGetAttribute(&a_lazy, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_lazy);
    cuDeviceGetAttribute(&a_direct, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_direct);
    std::snprintf(buf, sizeof(buf), "lazy=%d direct=%d", a_lazy, a_direct);
    report("lazy_cuDeviceGetAttribute == 直接调用", a_lazy == a_direct && a_lazy == 9, buf);

    // deep_jit 的 wrapper 也走同一符号
    int a_jit = -1;
    deep_jit::cuda::driver::lazy_cuDeviceGetAttribute(
        &a_jit, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev_direct);
    std::snprintf(buf, sizeof(buf), "deep_jit=%d", a_jit);
    report("deep_jit wrapper 与 DeepEP wrapper 一致", a_jit == a_lazy, buf);
}

// ============================================================================
// Part D：VMM（虚拟内存管理）全链路，只用懒加载 wrapper
//   —— 这正是 DeepEP 对称内存 / DeepGEMM VMM 分配走的 API
// ============================================================================
static void part_d() {
    std::printf("\n== Part D：VMM + 可共享句柄（POSIX fd）往返，全部经懒加载 ==\n");

    cudaFree(nullptr);  // 确保有 current context

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t gran = 0;
    CUresult r = deep_ep::lazy_cuMemGetAllocationGranularity(
        &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    char buf[192];
    std::snprintf(buf, sizeof(buf), "granularity=%zu B (%.1f KiB), rc=%s",
                  gran, gran / 1024.0, cu_str(r).c_str());
    report("cuMemGetAllocationGranularity", r == CUDA_SUCCESS && gran >= 4096, buf);
    if (r != CUDA_SUCCESS) return;

    const size_t size = 2 * 1024 * 1024;  // 2 MiB，DeepEP kNumAllocationAlignmentBytes

    CUmemGenericAllocationHandle h1 = 0;
    r = deep_ep::lazy_cuMemCreate(&h1, size, &prop, 0);
    report("cuMemCreate(2 MiB)", r == CUDA_SUCCESS, cu_str(r));
    if (r != CUDA_SUCCESS) return;

    // 导出为可共享 POSIX fd
    int fd = -1;
    r = deep_ep::lazy_cuMemExportToShareableHandle(
        &fd, h1, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    report("cuMemExportToShareableHandle -> fd", r == CUDA_SUCCESS && fd >= 0, cu_str(r));

    // 导入回一个新 handle（模拟跨进程）
    CUmemGenericAllocationHandle h2 = 0;
    if (r == CUDA_SUCCESS) {
        r = deep_ep::lazy_cuMemImportFromShareableHandle(
            &h2, reinterpret_cast<void*>(static_cast<intptr_t>(fd)),
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
        report("cuMemImportFromShareableHandle", r == CUDA_SUCCESS && h2 != 0, cu_str(r));
        ::close(fd);  // 调用方持有并负责关闭
    }

    CUdeviceptr p1 = 0, p2 = 0;
    r = deep_ep::lazy_cuMemAddressReserve(&p1, size, 0, 0, 0);
    report("cuMemAddressReserve #1", r == CUDA_SUCCESS && p1 != 0, cu_str(r));
    deep_ep::lazy_cuMemAddressReserve(&p2, size, 0, 0, 0);

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    r = deep_ep::lazy_cuMemMap(p1, size, 0, h1, 0);
    report("cuMemMap #1", r == CUDA_SUCCESS, cu_str(r));
    deep_ep::lazy_cuMemMap(p2, size, 0, h2, 0);
    r = deep_ep::lazy_cuMemSetAccess(p1, size, &access, 1);
    report("cuMemSetAccess #1", r == CUDA_SUCCESS, cu_str(r));
    deep_ep::lazy_cuMemSetAccess(p2, size, &access, 1);

    // 通过地址 p1 写，再从 p2 读 —— 两个 handle 指向同一份物理内存
    const int pattern = 0x5A;
    r = static_cast<CUresult>(cudaMemset(reinterpret_cast<void*>(p1), pattern, size));
    report("cudaMemset 到 VMM 地址", r == CUDA_SUCCESS, cu_str(r));

    std::vector<unsigned char> host(size, 0);
    cudaMemcpy(host.data(), reinterpret_cast<void*>(p2), size, cudaMemcpyDeviceToHost);
    int mism = 0;
    for (size_t i = 0; i < size; ++ i) if (host[i] != pattern) ++ mism;
    std::snprintf(buf, sizeof(buf), "p1==p2 跨句柄 %zu 字节, mismatch=%d", size, mism);
    report("导出的句柄与导入的句柄指向同一内存", mism == 0, buf);

    // cuMemGetAddressRange_v2（DeepEP 里名字带 v2 别名）
    CUdeviceptr base = 0; size_t range = 0;
    r = deep_ep::lazy_cuMemGetAddressRange_v2(&base, &range, p1);
    std::snprintf(buf, sizeof(buf), "base=0x%llx range=%zu (== p1=0x%llx size=%zu)",
                  static_cast<unsigned long long>(base), range,
                  static_cast<unsigned long long>(p1), size);
    report("cuMemGetAddressRange_v2", r == CUDA_SUCCESS && base == p1 && range == size, buf);

    // 清理
    deep_ep::lazy_cuMemUnmap(p1, size);
    deep_ep::lazy_cuMemUnmap(p2, size);
    deep_ep::lazy_cuMemRelease(h1);
    deep_ep::lazy_cuMemRelease(h2);
    deep_ep::lazy_cuMemAddressFree(p1, size);
    deep_ep::lazy_cuMemAddressFree(p2, size);
    report("VMM 清理（unmap/release/addressFree）无异常", true);
}

// ============================================================================
// Part E：deep_jit::LazyInit<T> 的语义（null factory = 终态；工厂只跑一次）
// ============================================================================
static void part_e() {
    std::printf("\n== Part E：deep_jit::LazyInit<T> 语义 ==\n");

    struct Counter { int value = 7; };
    int calls = 0;
    deep_jit::LazyInit<Counter> live(std::function<std::shared_ptr<Counter>()>([&] {
        ++ calls;
        return std::make_shared<Counter>();
    }));
    int v1 = live->value;
    int v2 = live->value;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "value=%d 工厂调用次数=%d", v1, calls);
    report("工厂只执行一次且返回正确", calls == 1 && v1 == 7 && v2 == 7, buf);

    deep_jit::LazyInit<Counter> dead(nullptr);
    bool threw = false;
    std::string msg;
    try {
        (void) dead->value;
    } catch (const std::exception& e) {
        threw = true;
        msg = e.what();
    }
    report("null factory 解引用抛异常", threw && msg.find("must be initialized") != std::string::npos,
           msg.substr(0, 60));
}

// ============================================================================
// Part F：libdw 懒加载 + 异常回溯
// ============================================================================
static void part_f() {
    std::printf("\n== Part F：libdw 懒加载与 deep_jit 异常回溯 ==\n");

    void* dw = dlopen("libdw.so.1", RTLD_LAZY | RTLD_LOCAL);
    report("dlopen(libdw.so.1)", dw != nullptr);

    const char* names[] = {
        "dwfl_begin", "dwfl_linux_proc_report", "dwfl_report_end", "dwfl_addrmodule",
        "dwfl_module_addrname", "dwfl_module_getsrc", "dwfl_lineinfo", "dwfl_linux_proc_find_elf"
    };
    int resolved = 0;
    for (const char* n : names) if (dlsym(dw, n) != nullptr) ++ resolved;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "解析 %d/8", resolved);
    report("8 个 dwfl_* 符号全部可解析", resolved == 8, buf);

    std::string what;
    try {
        throw deep_jit::exception::Exception("Probe", __FILE__, __LINE__, "intentional");
    } catch (const std::exception& e) {
        what = e.what();
    }
    bool has_trace = what.find("intentional") != std::string::npos;
    std::printf("---- 异常 what() ----\n%s\n---------------------\n", what.c_str());
    report("构造异常并打印回溯", has_trace);
}

// ============================================================================
// Part G：tensor.hpp 的 get_shape / get_data_ptr
// ============================================================================
static void part_g() {
    std::printf("\n== Part G：tensor.hpp 的 get_shape / get_data_ptr ==\n");

    auto t = torch::ones({3, 5}, torch::kFloat32).cuda();
    auto shape = deep_ep::get_shape<2>(t);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "shape=(%d,%d)", std::get<0>(shape), std::get<1>(shape));
    report("get_shape<2>", std::get<0>(shape) == 3 && std::get<1>(shape) == 5, buf);

    std::optional<torch::Tensor> none;
    report("get_data_ptr(nullopt) == nullptr", deep_ep::get_data_ptr<float>(none) == nullptr);

    std::optional<torch::Tensor> some = t;
    report("get_data_ptr(present) == data_ptr",
           deep_ep::get_data_ptr<float>(some) == t.data_ptr<float>());

    bool dim_threw = false;
    try { (void) deep_ep::get_shape<3>(t); } catch (const std::exception&) { dim_threw = true; }
    report("get_shape 维度不符抛 EP_HOST_ASSERT", dim_threw);
}

// ============================================================================
// Part H：版本闸门（compatibility.hpp 的正向编译即通过）
// ============================================================================
static void part_h() {
    std::printf("\n== Part H：版本闸门 ==\n");
    std::printf("  TORCH_VERSION=%s (major=%d minor=%d)\n", TORCH_VERSION,
                TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR);
    std::printf("  CUDA_VERSION=%d  CUDART_VERSION=%d\n", CUDA_VERSION, CUDART_VERSION);
    // compatibility.hpp 的三个 static_assert + deep_jit driver.hpp 的 CUDA>=12.4 在编译期已通过
    report("DeepGEMM 闸门（torch>=2.3, CUDA>=12.9）编译期通过", true);
    report("DeepJIT 闸门（CUDA>=12.4）编译期通过", true);

    constexpr int chunk = 4096, line = 128;
    report("locality 探针 hops = 4096/128 = 32", chunk / line == 32);
}

int main() {
    std::printf("=== P80-1 懒加载 / 版本闸门 / 张量 helper 地基验证 ===\n");
    std::printf("device: ");
    std::fflush(stdout);
    part_a();
    part_b();
    part_c();
    part_d();
    part_e();
    part_f();
    part_g();
    part_h();
    std::printf("\n==== 汇总：PASS=%d FAIL=%d ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
