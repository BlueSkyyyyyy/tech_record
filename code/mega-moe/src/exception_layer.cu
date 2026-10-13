// P116-1 实验：两库共用的「错误 / 断言层」——DeepEP `common/exception.cuh` 与
// DeepGEMM `deep_gemm/common/exception.cuh`（device）+ `csrc/utils/exception.hpp`（host）。
//
// 为什么做这个：专题前 115 篇把 dispatch / combine / mega moe / 各 grouped GEMM 的每一行
// `EP_HOST_ASSERT` / `DG_HOST_ASSERT` / `CUDA_RUNTIME_CHECK` / `NCCL_CHECK` 都当作「空气」
// ——它们是每次 kernel 发射前/后都踩的地板，却从没有一篇把这块地板本身拆开。本实验**直接
// #include 两库真实头文件**，把它们抛出的异常、消息格式、宏分派、设备端 trap 全部跑一遍。
//
// 关键结论（真机核对）：
//   * `EPException`/`DGException` 都只继承 `std::exception`（无自定义析构、无 override what 之外的东西）
//     → pybind11 默认翻译器把它们变成 Python `RuntimeError`（另见 exception_bridge.cpp）。
//   * 消息格式不同：EP=`"<name> exception (file:line): msg"`，DG=`"<name> error (file:line): msg"`。
//   * 宏分派键不同：EP 用 `#ifdef __CUDA_ARCH__`；DG 用 `DG_IN_CUDA_COMPILATION`（nvcc 全程定义）
//     → 这带来一个真实陷阱：**在 nvcc 编译的 host 代码里用 `DG_UNIFIED_ASSERT` 会展开成
//     `asm("trap;")`，汇编期报 "no such instruction: trap"**。EP 的 `EP_UNIFIED_ASSERT` 在
//     host 下正确落到 `EP_HOST_ASSERT`。
//
// 编译：nvcc -arch=sm_90 -std=c++17 -O2 <includes> src/exception_layer.cu -o /tmp/exception_layer \
//         -L/tmp/nccl229/lib -lnccl -lcublasLt -lcublas -lcuda -ldl
// 运行：/tmp/exception_layer {host|dev_ep_assert|dev_ep_unified|dev_dg_assert|dev_dg_traponly|dev_dg_unified}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

#include <nccl.h>                                  // 必须在 exception.cuh 之前（NCCL_CHECK 引用 nccl*）
#include <deep_ep/common/exception.cuh>            // EPException / EP_* 宏
#include "utils/lazy_driver.hpp"                   // lazy_cuDeviceGet / lazy_cuGetErrorName（CUDA_DRIVER_CHECK 依赖）
#include <deep_gemm/common/exception.cuh>          // DG_DEVICE_ASSERT / DG_UNIFIED_ASSERT / DG_STATIC_ASSERT
#include "utils/exception.hpp"                     // DGException / DG_HOST_ASSERT / DG_CUDA_RUNTIME_CHECK / DG_CUBLASLT_CHECK

using namespace deep_gemm;                         // DG_HOST_ASSERT 展开出的 `throw DGException(...)` 需要可见

static int g_pass = 0, g_fail = 0;
static void check(const char* name, bool ok) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name);
    ok ? ++g_pass : ++g_fail;
}

// 在 deep_ep 命名空间里触发 CUDA_DRIVER_CHECK（宏内引用 lazy_cuGetErrorName/String，是 deep_ep 的成员）
namespace deep_ep {
static void trigger_driver_check() {
    CUdevice dev;
    CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&dev, 999));
}
}  // namespace deep_ep

// ----------------------------------------------------------------------------------------------
// 设备端 kernel：每个只负责触发一种 device assert，供 host 侧单独进程隔离观测
// ----------------------------------------------------------------------------------------------
__global__ void k_ep_device_assert()  { EP_DEVICE_ASSERT(1 == 2); }
__global__ void k_ep_unified()        { EP_UNIFIED_ASSERT(1 == 2); }   // device pass: __CUDA_ARCH__ 定义 → EP_DEVICE_ASSERT
__global__ void k_dg_device_assert()  { DG_DEVICE_ASSERT(1 == 2); }
__global__ void k_dg_traponly()       { DG_TRAP_ONLY_DEVICE_ASSERT(1 == 2); }
__global__ void k_dg_unified()        { DG_UNIFIED_ASSERT(1 == 2); }   // device pass: DG_IN_CUDA_COMPILATION → DG_DEVICE_ASSERT

// ----------------------------------------------------------------------------------------------
// host 模式：所有 host 侧检查
// ----------------------------------------------------------------------------------------------
static int run_host() {
    std::printf("== A. EPException 消息格式 / 类层次 ==\n");
    {
        EPException e("Assertion", "foo.cuh", 42, "cond failed");
        std::string m = e.what();
        std::printf("   what() = `%s`\n", m.c_str());
        check("A1 EPException.what() 含 name/文件/行/原因",
              m.find("Assertion") == 0 && m.find("foo.cuh:42") != std::string::npos &&
              m.find("cond failed") != std::string::npos && m.find("exception") != std::string::npos);
        check("A2 EPException 是 std::exception 子类", dynamic_cast<const std::exception*>(&e) != nullptr);
    }
    {
        try { throw EPExceptionWithLineInfo("Dispatch CPU wait", "buffer#0"); }
        catch (const std::exception& e) {
            std::string m = e.what();
            std::printf("   EPExceptionWithLineInfo = `%s`\n", m.c_str());
            check("A3 EPExceptionWithLineInfo 注入 __FILE__/__LINE__ 且 name=Dispatch CPU wait",
                  m.find("Dispatch CPU wait") == 0 && m.find(".cu:") != std::string::npos);
        }
    }

    std::printf("== B. EP_HOST_ASSERT / EP_HOST_UNREACHABLE ==\n");
    {
        bool thrown = false;
        try { EP_HOST_ASSERT(2 + 2 == 4); } catch (...) { thrown = true; }
        check("B1 EP_HOST_ASSERT(true) 不抛", !thrown);
        try { EP_HOST_ASSERT(2 + 2 == 5); }
        catch (const EPException& e) { thrown = true; check("B2 EP_HOST_ASSERT(false) 抛 EPException（含 #cond 文本）",
              std::string(e.what()).find("2 + 2 == 5") != std::string::npos); }
        if (!thrown) check("B2 EP_HOST_ASSERT(false) 抛 EPException", false);
        try { EP_HOST_UNREACHABLE("should never get here"); check("B3 EP_HOST_UNREACHABLE 抛异常", false); }
        catch (const EPException& e) { check("B3 EP_HOST_UNREACHABLE 抛 EPException",
              std::string(e.what()).find("should never get here") != std::string::npos); }
    }

    std::printf("== C. EP_UNIFIED_ASSERT 的 host 分派 ==\n");
    {
        bool thrown = false;
        try { EP_UNIFIED_ASSERT(false); } catch (const EPException&) { thrown = true; }
        check("C1 host 下 EP_UNIFIED_ASSERT → EP_HOST_ASSERT（抛 EPException 而非 trap）", thrown);
#ifdef __CUDA_ARCH__
        std::printf("   (host 段 __CUDA_ARCH__ 意外已定义)\n");
#endif
    }

    std::printf("== D. CUDA_RUNTIME_CHECK（DeepEP）==\n");
    {
        bool thrown = false;
        try { CUDA_RUNTIME_CHECK(cudaSetDevice(9999)); }
        catch (const EPException& e) { thrown = true; check("D1 CUDA_RUNTIME_CHECK 把 cuda 错误翻成 EPException（name=CUDA runtime）",
              std::string(e.what()).find("CUDA runtime") == 0 &&
              std::string(e.what()).find("invalid device ordinal") != std::string::npos); }
        if (!thrown) check("D1 CUDA_RUNTIME_CHECK 抛 EPException", false);
        cudaGetLastError();  // 清掉 sticky
    }

    std::printf("== E. CUDA_DRIVER_CHECK（DeepEP，走 lazy_* 符号）==\n");
    {
        cuInit(0);  // 让驱动已初始化，杜绝 NOT_INITIALIZED 噪声
        bool thrown = false;
        try { deep_ep::trigger_driver_check(); }
        catch (const EPException& e) { thrown = true;
            std::string m = e.what();
            std::printf("   %s\n", m.c_str());
            check("E1 CUDA_DRIVER_CHECK 翻成 EPException（name=CUDA driver，含 lazy_cuGetError* 文本）",
                  m.find("CUDA driver") == 0 && m.find("CUDA_ERROR_INVALID") != std::string::npos); }
        if (!thrown) check("E1 CUDA_DRIVER_CHECK 抛 EPException", false);
    }

    std::printf("== F. NCCL_CHECK（DeepEP）==\n");
    {
        bool thrown = false;
        try { NCCL_CHECK(ncclGetVersion(nullptr)); }
        catch (const EPException& e) { thrown = true; check("F1 NCCL_CHECK 翻成 EPException（name=NCCL）",
              std::string(e.what()).find("NCCL") == 0); }
        if (!thrown) check("F1 NCCL_CHECK 抛 EPException", false);
    }

    std::printf("== G. EP_STATIC_ASSERT（编译期）==\n");
    {
        EP_STATIC_ASSERT(sizeof(int) >= 4, "int must be >= 4 bytes");
        check("G1 EP_STATIC_ASSERT(true) 正常编译通过", true);
    }

    std::printf("== H. DGException 消息格式 / 类层次 ==\n");
    {
        DGException e("Assertion", "bar.cuh", 7, "nope");
        std::string m = e.what();
        std::printf("   what() = `%s`\n", m.c_str());
        check("H1 DGException.what() 用 `error` 而非 `exception`",
              m.find("Assertion error") == 0 && m.find("bar.cuh:7") != std::string::npos && m.find("nope") != std::string::npos);
        check("H2 DGException 是 std::exception 子类", dynamic_cast<const std::exception*>(&e) != nullptr);
    }
    {
        bool thrown = false;
        try { DG_HOST_ASSERT(1 == 2); }
        catch (const DGException& e) { thrown = true; check("H3 DG_HOST_ASSERT 抛 DGException（含 #cond）",
              std::string(e.what()).find("1 == 2") != std::string::npos); }
        if (!thrown) check("H3 DG_HOST_ASSERT 抛 DGException", false);
        try { DG_HOST_UNREACHABLE("dg unreachable"); check("H4 DG_HOST_UNREACHABLE 抛异常", false); }
        catch (const DGException& e) { check("H4 DG_HOST_UNREACHABLE 抛 DGException",
              std::string(e.what()).find("dg unreachable") != std::string::npos); }
    }

    std::printf("== I. DG_CUDA_RUNTIME_CHECK / DG_CUBLASLT_CHECK（DeepGEMM）==\n");
    {
        bool t1 = false;
        try { DG_CUDA_RUNTIME_CHECK(cudaSetDevice(9999)); }
        catch (const DGException& e) { t1 = true; check("I1 DG_CUDA_RUNTIME_CHECK 翻成 DGException（name=CUDA runtime）",
              std::string(e.what()).find("CUDA runtime") == 0); }
        if (!t1) check("I1 DG_CUDA_RUNTIME_CHECK 抛 DGException", false);
        cudaGetLastError();
        bool t2 = false;
        try { DG_CUBLASLT_CHECK(cublasLtCreate(nullptr)); }
        catch (const DGException& e) { t2 = true; check("I2 DG_CUBLASLT_CHECK 翻成 DGException（name=cuBLASLt）",
              std::string(e.what()).find("cuBLASLt") == 0); }
        if (!t2) check("I2 DG_CUBLASLT_CHECK 抛 DGException", false);
    }

    std::printf("== J. DG_UNIFIED_ASSERT 的 host 分派键（真实陷阱）==\n");
    {
#ifdef DG_IN_CUDA_COMPILATION
        check("J1 nvcc 编译下 DG_IN_CUDA_COMPILATION 已定义 → DG_UNIFIED_ASSERT 走 DG_DEVICE_ASSERT", true);
        std::printf("   → 因此源文件里**不能**在 host 代码用 DG_UNIFIED_ASSERT（会展开出 asm(\"trap;\")，汇编失败）\n");
#else
        check("J1 DG_IN_CUDA_COMPILATION 未定义 → DG_UNIFIED_ASSERT 走 DG_HOST_ASSERT", true);
#endif
    }

    std::printf("\nHOST RESULT: PASS=%d FAIL=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

// ----------------------------------------------------------------------------------------------
// 设备模式：每个独立进程跑一个 kernel，避免 trap 的 sticky 错误互相污染
// ----------------------------------------------------------------------------------------------
template <typename Launch>
static int run_device(const char* name, Launch launch) {
    launch();
    cudaError_t e = cudaDeviceSynchronize();
    std::printf("DEVICE %-18s sync=%s (%s)\n", name, cudaGetErrorName(e), cudaGetErrorString(e));
    cudaDeviceReset();
    return 0;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "host";
    if (mode == "host")           return run_host();
    if (mode == "dev_ep_assert")  return run_device("EP_DEVICE_ASSERT",   [] { k_ep_device_assert<<<1, 1>>>(); });
    if (mode == "dev_ep_unified") return run_device("EP_UNIFIED_ASSERT",  [] { k_ep_unified<<<1, 1>>>(); });
    if (mode == "dev_dg_assert")  return run_device("DG_DEVICE_ASSERT",   [] { k_dg_device_assert<<<1, 1>>>(); });
    if (mode == "dev_dg_traponly")return run_device("DG_TRAP_ONLY",       [] { k_dg_traponly<<<1, 1>>>(); });
    if (mode == "dev_dg_unified") return run_device("DG_UNIFIED_ASSERT",  [] { k_dg_unified<<<1, 1>>>(); });
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
}
