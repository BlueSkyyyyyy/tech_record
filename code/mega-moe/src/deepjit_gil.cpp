// P160-1 实验（pybind11）：证明 DeepJIT 的 GilScopedRelease（utils/gil.hpp）在从 Python
// 进入 C++ 时释放 GIL、且在 GIL 已释放/解释器未初始化时安全地「不释放」。
// 编译（.sh 里）：g++ -std=c++20 -O2 -shared -fPIC -I<pybind11>/include -I/usr/include/python3.12 \
//                    src/deepjit_gil.cpp -o /tmp/deepjit_gil$(python3-config --extension-suffix) -lpython3.12
#include <tuple>

#include <pybind11/pybind11.h>

#include <deep_jit/utils/gil.hpp>

namespace py = pybind11;

static_assert(!std::is_copy_constructible_v<deep_jit::GilScopedRelease>, "must be non-copyable");
static_assert(!std::is_move_constructible_v<deep_jit::GilScopedRelease>, "must be non-movable");

// 普通入口：Python 持 GIL 调用进来 → before=1；构造 GilScopedRelease 后 → inside=0；析构后复位。
static std::tuple<int, int, int> probe() {
    const int before = PyGILState_Check();
    int inside = -1;
    {
        deep_jit::GilScopedRelease gil_release;
        inside = PyGILState_Check();
    }
    const int after = PyGILState_Check();
    return {before, inside, after};
}

// GIL 已在外面被释放（call_guard）：before=0，GilScopedRelease 不应重复释放。
static std::tuple<int, int> probe_no_gil() {
    int before = -1, inside = -1;
    {
        deep_jit::GilScopedRelease gil_release;
        before = PyGILState_Check();
        inside = PyGILState_Check();
    }
    return {before, inside};
}

PYBIND11_MODULE(deepjit_gil, m) {
    m.doc() = "P160-1: DeepJIT GilScopedRelease probe";
    m.def("probe", &probe, "return (before, inside, after) PyGILState_Check()");
    m.def("probe_no_gil", &probe_no_gil, py::call_guard<py::gil_scoped_release>(),
          "call with GIL pre-released");
}
