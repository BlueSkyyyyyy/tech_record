// P7-111: localize the mechanism behind the write-side `dram__bytes_write`
// scatter that §99.5 (P7-96) and §102.4 (P7-99) left open:
//   - §99.5: `b2_mma_probe.cu` bf16 vs fp32 at V=256 differ by +35 MB in
//     `dram__bytes_write`, while `l1tex__..._op_st` (store sectors) are bitwise
//     identical and `lts__t_sectors_op_write` is even *lower* for bf16; the
//     explanation offered is "same bytes, different L2->DRAM eviction share".
//   - §102.4: the same config across ncu replays differs by ~19 MB.
//
// Hypothesis under test: `dram__bytes_write` measured inside a single ncu
// kernel window is not the output size but *the fraction of dirty L2 lines that
// get flushed to DRAM before the window closes*, which is set by
//   (a) L2 eviction pressure during the kernel (a concurrent read stream), and
//   (b) whether the output working set fits in L2 at all.
// The store *instructions* are identical in every variant, so
// `l1tex__t_sectors_pipe_lsu_mem_global_op_st` (S_L1) must stay constant while
// `dram__bytes_write` moves.
//
// Self-contained microbenchmark: one kernel, four read/write orderings, output
// footprint `nout` and read footprint `nread` swept from the driver
// (`harness/probe_l2_write_evict.py`), profiled by
// `harness/probe_l2_write_evict_ncu.sh` (nvcc hangs under ncu in this container
// (P7-16) -> pre-build outside ncu).
//
//   MODE 0 : store only
//   MODE 1 : read all, then store all
//   MODE 2 : store all, then read all
//   MODE 3 : interleaved (read half, store half, ...)
//
// Every store is a full-128B-line `float4` write with a grid-stride loop that
// covers the output exactly once.
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float4 ldv4(const float4* __restrict__ p) {
    return *p;
}

__device__ __forceinline__ float read_stream(const float* __restrict__ r, long n,
                                             int tid, int nth, float& sink) {
    float acc = 0.f;
    // grid-stride over float4 chunks, fully covering every 128B line.
    const long nv = n >> 2;
    for (long i = tid; i < nv; i += nth) {
        float4 v = ldv4(reinterpret_cast<const float4*>(r) + i);
        acc += v.x + v.y + v.z + v.w;
    }
    sink += acc;
    return acc;
}

template <int MODE>
__global__ void writeKernel(const float* __restrict__ rbuf, float* __restrict__ out,
                            long nout, long nread, float* __restrict__ sinkbuf) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int nth = gridDim.x * blockDim.x;
    const long nv = nout >> 2;
    float sink = 0.f;

    if (MODE == 0) {
        for (long i = tid; i < nv; i += nth) {
            reinterpret_cast<float4*>(out)[i] = make_float4(1.f, 2.f, 3.f, 4.f);
        }
    } else if (MODE == 1) {
        read_stream(rbuf, nread, tid, nth, sink);
        for (long i = tid; i < nv; i += nth) {
            reinterpret_cast<float4*>(out)[i] = make_float4(1.f, 2.f, 3.f, 4.f);
        }
    } else if (MODE == 2) {
        for (long i = tid; i < nv; i += nth) {
            reinterpret_cast<float4*>(out)[i] = make_float4(1.f, 2.f, 3.f, 4.f);
        }
        read_stream(rbuf, nread, tid, nth, sink);
    } else {  // MODE 3: interleave -- store one float4, read one float4
        long i = tid, j = tid;
        const long nvr = nread >> 2;
        while (i < nv || j < nvr) {
            if (i < nv) {
                reinterpret_cast<float4*>(out)[i] = make_float4(1.f, 2.f, 3.f, 4.f);
                i += nth;
            }
            if (j < nvr) {
                float4 v = ldv4(reinterpret_cast<const float4*>(rbuf) + j);
                sink += v.x + v.y + v.z + v.w;
                j += nth;
            }
        }
    }
    if (sinkbuf) sinkbuf[tid] = sink;  // keep the read live
}

#define DISPATCH(M)                                                              \
    if (mode == (M)) {                                                           \
        writeKernel<M><<<grid, block>>>(rbuf, out, nout, nread, sink);           \
        return (int)cudaGetLastError();                                          \
    }

extern "C" int probe_launch(int mode, const float* rbuf, float* out,
                            long nout, long nread, float* sink,
                            int grid, int block) {
    DISPATCH(0) DISPATCH(1) DISPATCH(2) DISPATCH(3)
    return -1;
}
