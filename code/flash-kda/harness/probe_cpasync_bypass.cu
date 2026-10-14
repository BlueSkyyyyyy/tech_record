// P7-110: localize §51.7's "cp.async L1-bypass path reads 1.5x more L2 sectors"
// mechanism.  §51.4 showed that the K2-shaped mma CUDA B2 GEMM core's `cp.async`
// (LDGSTS.E.BYPASS.128, 16 B/thread) reads 59.77 M L2 srcunit sectors while the
// *identical-op, identical-16B, identical-pattern* plain `ld.global.v4` reads
// exactly the DRAM sector count 39.85 M (1.0x).  It confirmed the amplification
// is a `.BYPASS`/cp.async *path* property (not the 16 B width, not the "4 rows x
// 128 B per warp" pattern) but left the sector-level cause open, and named the
// experiment: compare `cp.async.ca` (allocates L1) against `cp.async.cg`
// (L1-bypass) at the same width/pattern.
//
// This probe is a self-contained microbenchmark isolating only the *read path*:
// every variant issues the same 512 B / warp / iteration access pattern over the
// same buffer, only the load instruction differs:
//   MODE 0 : cp.async.cg.shared.global 16 B   (L1-bypass, = LDGSTS.BYPASS.128)
//   MODE 1 : cp.async.ca.shared.global 16 B   (L1-allocating)
//   MODE 2 : cp.async.ca.shared.global  8 B x2
//   MODE 3 : cp.async.ca.shared.global  4 B x4
//   MODE 4 : ld.global.v4.f32         16 B   (L1-allocating)
//   MODE 5 : ld.global.cg.v4.f32      16 B   (L2-only, L1-bypass)
//   MODE 6 : ld.global.f32 scalar      4 B x4
//   PAT  0 : contiguous 128 floats (512 B) per warp
//   PAT  1 : 4 rows x 32 floats (128 B each), row pitch = `pitch` floats
//
// Built as a .so, driven by `harness/probe_cpasync_bypass.py` (ctypes); profiled
// by `harness/probe_cpasync_bypass_ncu.sh`.  NOTE: nvcc hangs under ncu in this
// container (P7-16) -> pre-build outside ncu.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

template <int MODE>
__device__ __forceinline__ void do_read(const float* __restrict__ base,
                                        float* __restrict__ sm, float& acc) {
    if (MODE == 0) {
        unsigned sa = smem_u32(sm);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(base)
                     : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        acc += sm[0]; acc += sm[1]; acc += sm[2]; acc += sm[3];
    } else if (MODE == 1) {
        unsigned sa = smem_u32(sm);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(base)
                     : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        acc += sm[0]; acc += sm[1]; acc += sm[2]; acc += sm[3];
    } else if (MODE == 2) {
        unsigned sa = smem_u32(sm);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(sa + 0), "l"(base)
                     : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(sa + 8), "l"(base + 2)
                     : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        acc += sm[0]; acc += sm[1]; acc += sm[2]; acc += sm[3];
    } else if (MODE == 3) {
        unsigned sa = smem_u32(sm);
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(sa + 0), "l"(base)
                     : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(sa + 4), "l"(base + 1)
                     : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(sa + 8), "l"(base + 2)
                     : "memory");
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(sa + 12), "l"(base + 3)
                     : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        acc += sm[0] + sm[1] + sm[2] + sm[3];
    } else if (MODE == 4) {
        float4 v = *reinterpret_cast<const float4*>(base);
        acc += v.x + v.y + v.z + v.w;
    } else if (MODE == 5) {
        float a, b, c, d;
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                     : "=f"(a), "=f"(b), "=f"(c), "=f"(d)
                     : "l"(base));
        acc += a + b + c + d;
    } else {  // MODE 6
        acc += base[0] + base[1] + base[2] + base[3];
    }
}

template <int MODE, int PAT>
__global__ void probeKernel(const float* __restrict__ x, float* __restrict__ out,
                            long nunits, int iters, long pitch) {
    extern __shared__ float sm[];
    const int lane = threadIdx.x & 31;
    const int warp = (int)((blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5);
    const int nwarps = (int)((gridDim.x * (long)blockDim.x) >> 5);
    float* my = sm + threadIdx.x * 4;
    float acc = 0.f;
    if (nunits <= 0) return;
    for (int it = 0; it < iters; ++it) {
        for (long u = warp; u < nunits; u += nwarps) {
            const float* base;
            if (PAT == 0) {
                base = x + u * 128 + lane * 4;  // 32 lanes x 16B = 512B contiguous
            } else {
                long row = u * 4 + (lane >> 3);
                base = x + row * pitch + (lane & 7) * 4;
            }
            do_read<MODE>(base, my, acc);
        }
    }
    out[blockIdx.x * (long)blockDim.x + threadIdx.x] = acc;
}

#define DISPATCH(M, P)                                                        \
    if (mode == (M) && pat == (P)) {                                          \
        probeKernel<M, P><<<grid, block, (size_t)block * 4 * sizeof(float)>>>( \
            x, out, nunits, iters, pitch);                                    \
        return (int)cudaGetLastError();                                       \
    }

extern "C" int probe_launch(int mode, int pat, const float* x, float* out,
                            long nunits, int iters, long pitch, int grid, int block) {
    // PAT 0 consumes x up to nunits*128 floats; PAT 1 up to nunits*4*pitch floats.
    DISPATCH(0, 0) DISPATCH(0, 1)
    DISPATCH(1, 0) DISPATCH(1, 1)
    DISPATCH(2, 0) DISPATCH(2, 1)
    DISPATCH(3, 0) DISPATCH(3, 1)
    DISPATCH(4, 0) DISPATCH(4, 1)
    DISPATCH(5, 0) DISPATCH(5, 1)
    DISPATCH(6, 0) DISPATCH(6, 1)
    return -1;
}
