"""Print exact `sizeof(SharedStorageK2<...>)` for the occupancy variants.

Uses the cond-patched header (ReserveFP32 template arg).  A plain build without
the patch is identical to ReserveFP32=true.
"""
import os
from torch.utils.cpp_extension import load_inline

BASE = "/tmp/kda_occ/src_cond/csrc"
CUTLASS = os.path.expanduser("~/github/FlashKDA/cutlass")

cpp = "std::vector<long> sizes();"
cu = r"""
#include "fwd_kernel2.cuh"
#include <vector>
std::vector<long> sizes() {
    using L = K2Layouts<128, 16>;
    return {
        (long)sizeof(SharedStorageK2<L, 3, 2, true>),   // orig / plain (3,2)
        (long)sizeof(SharedStorageK2<L, 2, 1, true>),   // plain (2,1)
        (long)sizeof(SharedStorageK2<L, 3, 2, false>),  // cond (3,2), bf16
        (long)sizeof(SharedStorageK2<L, 2, 2, false>),  // cond (2,2), bf16
        (long)sizeof(SharedStorageK2<L, 2, 1, false>),  // cond (2,1), bf16
        (long)sizeof(SharedStorageK2<L, 1, 2, false>),  // cond (1,2), bf16
    };
}
"""
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
mod = load_inline(
    name="kda_smem_probe",
    cpp_sources=[cpp],
    cuda_sources=[cu],
    functions=["sizes"],
    extra_include_paths=[os.path.join(CUTLASS, "include"),
                         os.path.join(CUTLASS, "examples", "common"),
                         os.path.join(CUTLASS, "tools", "util", "include"),
                         BASE, os.path.join(BASE, "smxx")],
    extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "--expt-extended-lambda"],
    verbose=False,
)
labels = ["orig/plain (3,2)", "plain (2,1)", "cond (3,2) bf16", "cond (2,2) bf16",
          "cond (2,1) bf16", "cond (1,2) bf16"]
smem_per_sm = 233472   # H100 max shared per SM (bytes)
driver_smem = 1024     # "Driver Shared Memory Per Block" reported by ncu
print(f"{'variant':20s} {'smem_B':>8s} {'+driver':>8s} {'CTA/SM(analytic)':>16s}")
for lab, b in zip(labels, mod.sizes()):
    per = b + driver_smem
    ctas = smem_per_sm // per
    print(f"{lab:20s} {b:8d} {per:8d} {ctas:16d}")
print("\nNote: ncu 'Block Limit Shared Mem' == CTA/SM(analytic) (driver smem included);")
print("cond (2,2) is 77,184+1024 = 78,208 B -> 3*78,208 = 234,624 > 233,472 => still 2/SM.")
