// P116-1 实验：把两库真实的 `EPException` / `DGException` 抛进 Python，验证 pybind11 的
// 默认异常翻译器把它们变成 `RuntimeError`（两库都没有显式 `py::register_exception`）。
//
// 由 harness/exp_ep_dg_exception.py 通过 torch.utils.cpp_extension.load 编译加载。
// 直接 #include 两库真实头文件（DeepEP common/exception.cuh + DeepGEMM csrc/utils/exception.hpp）。

#include <torch/extension.h>

#include <deep_ep/common/exception.cuh>   // EPException / EP_HOST_ASSERT / EPExceptionWithLineInfo
#include "utils/exception.hpp"            // DeepGEMM DGException / DG_HOST_ASSERT / DG_HOST_UNREACHABLE

using namespace deep_gemm;                // DG_HOST_ASSERT 展开的 `throw DGException(...)` 需要可见

static void throw_ep_assert()   { EP_HOST_ASSERT(false && "ep assert from bridge"); }
static void throw_ep_line()     { throw EPExceptionWithLineInfo("Bridge", "line-info payload"); }
static void throw_ep_unreach()  { EP_HOST_UNREACHABLE("ep unreachable from bridge"); }
static void throw_dg_assert()   { DG_HOST_ASSERT(false && "dg assert from bridge"); }
static void throw_dg_unreach()  { DG_HOST_UNREACHABLE("dg unreachable from bridge"); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("throw_ep_assert",  &throw_ep_assert);
    m.def("throw_ep_line",    &throw_ep_line);
    m.def("throw_ep_unreach", &throw_ep_unreach);
    m.def("throw_dg_assert",  &throw_dg_assert);
    m.def("throw_dg_unreach", &throw_dg_unreach);
}
