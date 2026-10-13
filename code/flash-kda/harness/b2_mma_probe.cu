// P7-46: the *second* slice of the K2-shaped CUDA port of the fused B2/B3 -- the
// B2 (`wy_dqkg`) V-reduction GEMM core, this time **keeping `mma.sync` tensor core**
// as §47.5/§48.4 require, and contrasting P7-44's dead-end SIMT version.
//
// P7-44 (`b2_core_probe.cu`) tested reading (b) of §46.6's "把 dq/dk/dw 累加器按 warp
// 分片常驻寄存器": plain SIMT fp32 FMA.  It lost 3-5x to the identical-op Triton
// kernel (which is itself only 1.25x above its DRAM floor), pinning the answer to
// reading (a): *register-sharded accumulators **alongside** tensor core* (B1 §19-24
// recipe).  P7-45 (§48) then showed the fused kernel's ~1.9x headroom is not bytes
// and not occupancy but **"big operands (h/dh) kept register-resident -> spill/issue
// starvation"**, and named the first action:
//
//     "把大操作数（尤其 h/dh）从寄存器保活里拿掉、只留 warp 分片的小累加器".
//
// This prototype is that action on the isolated dominant GEMMs of B2:
//   dq[t,k] = sum_v do[t,v]*h[k,v]     (do  @ b_h^T)
//   dk[t,k] = sum_v v_new[t,v]*dh[k,v] (v_new @ b_dh^T)
//   dw[t,k] = sum_v dv[t,v]*h[k,v]     (dv  @ b_h^T)
//   dgk[k]  = sum_v h[k,v]*dh[k,v]
//
// Structure (the K2 shape, per chunk `(i_t, i_bh)` with grid `(NT, B*H)`):
//   * block = 8 warps; warp `w` owns the **N (K) output columns** [w*16, w*16+16).
//   * `h/dh/do/vn/dv` are **streamed through smem in V-tiles** (`BVK`) via `cp.async`
//     double buffering -- **no `[V,K]` operand is kept register-resident**; each warp
//     only carries its small `[16,16]` fp32 accumulators (`dq/dk/dw`) + one `dgk` scalar.
//   * the contraction is done with raw `mma.sync.m16n8k8.tf32` (matches `tl.dot`'s
//     default tf32 on fp32 inputs), fed by direct smem reads of the A/B fragments.
//
// This file implements the GEMM core (like P7-44) so it is an apples-to-apples head-to-head
// vs the identical-op Triton kernel; the B2 non-GEMM pieces are added as separate opt-in
// segments: P7-52 `-DB2M_OUT` (output branch `dv2=beta*(A@dv)`) and P7-53 `-DB2M_DKGB`
// (`dkgb=A@(-dw)`).  P7-54/55/56/57 added the remaining `dA`/`db`/gate segments and P7-60
// merged them into one production launch (`-DB2M_PROD`); P7-63 folds the B3 `intra` diagonal
// into that production launch (`-DB2M_INTRA`).
//
// Build (inside kernel_lab):
//   nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared b2_mma_probe.cu -o /tmp/b2_mma.so
// Host entry `extern "C" b2mma_launch(...)`, called from check_bwd_b2_mma.py via ctypes.
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cuda_bf16.h>

#define DEV __device__ __forceinline__

#if B2M_NOASYNC
DEV void cp16(void* dst, const void* src) {
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
}
#else
DEV void cp16(void* dst, const void* src) { __pipeline_memcpy_async(dst, src, 16); }
#endif

// P7-81: `KD` (K = key / output-N dim) and `VD` (V = value / contraction dim) are
// independent compile-time sizes.  The K2-shaped **base GEMM core** (`dq/dk/dw/dgk`)
// already indexes the two dims consistently (K-dim outputs use `KD`, V-dim operands
// use `VD`; the TMA maps carry {VD,KD} separately), so the GEMM core is generic in
// K != V -- this macro pair just lets a build instantiate a non-128 V.  The six
// non-GEMM segments (`B2M_OUT`/`DB`/...) partition V by `BW = KD/NW`, so those still
// require `VD == KD` (they are off in the base GEMM core).
#ifndef B2M_KD
#define B2M_KD 128
#endif
#ifndef B2M_VD
#define B2M_VD 128
#endif
constexpr int KD = B2M_KD;   // K = D (output N)
constexpr int VD = B2M_VD;   // V = D (contraction K')
constexpr int BT = 16;    // chunk rows (M)
constexpr int NW = 8;     // warps
constexpr int BW = KD / NW;    // output columns per warp = KD / NW
static_assert(BW * NW == KD, "KD must equal NW*BW (warp output-column partition)");
// P7-82: the six non-GEMM segments split into a **K-axis** group (`DKGB`'s `dkgb`, the gate
// chain's `dqg/dkg/dg`, the `dgk` column reduction, `DA`'s `dw@kg^T`, the folded `intra`) that
// partitions K by `BW = KD/NW`, and a **V-axis** group (`OUT`'s `dv2/dvb`, `DA`'s `dv@v^T`,
// `DB`'s raw-v mirror) that partitions V.  The V-axis group gets its own warp width `BVW`, so
// the six segments are generic in K != V (P7-81 only covered the base GEMM core).
constexpr int BVW = VD / NW;   // V columns per warp, for the V-axis non-GEMM segments
constexpr int NBV = BVW / 8;   // mma `n=8` tiles per warp over V
static_assert(VD % NW == 0, "VD must be a multiple of NW (V-axis warp partition)");
static_assert(BVW % 8 == 0, "warp V-slice must be a multiple of the mma n=8");

#ifndef B2M_BVK
#define B2M_BVK 32
#endif
#ifndef B2M_NS
#define B2M_NS 2
#endif
// P7-47 diagnostics:
//   B2M_NODGK : skip the `dgk` column reduction entirely (dgk stored as 0) -- isolates
//               whether the 8.81M smem bank conflicts live in the dgk loop or in the
//               direct mma-fragment reads (§49.4.3).
//   B2M_DGKCF : conflict-free `dgk` reduction -- 8 kcols per warp instruction (see below).
#ifndef B2M_NODGK
#define B2M_NODGK 0
#endif
#ifndef B2M_DGKCF
#define B2M_DGKCF 0
#endif
// P7-48 diagnostics: localize §49.4.3's "L2 bytes +35% vs the identical-op Triton core"
// (3.03 vs 2.24 GB), which §50.5 left as the open lead.
//   B2M_NOHDH : skip the h/dh global cp.async (zero-fill smem) -> bytes side of h/dh.
//   B2M_NODOV : skip the do/vn/dv global cp.async (zero-fill smem).
//   B2M_VECST : write the three outputs with 64-bit (`float2`) stores instead of the
//               default two scalar 32-bit `STG.E` (candidate fix for half-sector stores).
#ifndef B2M_NOHDH
#define B2M_NOHDH 0
#endif
#ifndef B2M_NODOV
#define B2M_NODOV 0
#endif
#ifndef B2M_VECST
#define B2M_VECST 0
#endif
// P7-59: pairwise store helper -- under `B2M_VECST` emit one 64-bit `STG.E.64` (`float2`),
// else two 32-bit scalar stores.  Applied to the non-GEMM segment stores (`dv2/dvb`, `dkgb`,
// the gate chain's `dqg/dkg/dg`) which P7-52/53/56/57 left scalar.  Pairwise columns are
// contiguous and even-indexed (`KD=128`, col even) so the `float2` is 8 B aligned.
DEV void st2(float* p, float a, float b) {
#if B2M_VECST
    *reinterpret_cast<float2*>(p) = make_float2(a, b);
#else
    p[0] = a; p[1] = b;
#endif
}
//   B2M_NOASYNC : replace the 16B `cp.async` (`LDGSTS.E.BYPASS.128`) with a synchronous
//                 16B `float4` load+store -> same bytes, normal (L1-allocating) ld path.
//                 Isolates whether the 1.5x L2-read-sector amplification is specific to
//                 the cp.async/bypass path or to the access pattern itself.
#ifndef B2M_NOASYNC
#define B2M_NOASYNC 0
#endif
//   B2M_TMA (P7-49): stage the operands with `cp.async.bulk.tensor` (TMA) instead of
//                 `cp.async`.  §51.4 pinned the read-side +0.64 GB to the `cp.async`
//                 `.BYPASS` path (L1-bypass loses cross-lane 128 B coalescing) and named
//                 "读侧下一杠杆 = TMA 输入".  TMA issues per-tile descriptor copies off
//                 the LSU, into a SWIZZLE_128B smem layout (128 B = 32 fp32 inner rows),
//                 using a per-stage `mbarrier` (`complete_tx::bytes`) handshake.
//                 P7-50 tried BVK==64 with SWIZZLE_NONE (dense 256 B rows) -- bit-correct
//                 but 2.6-2.8x slower (dense 256 B stride = 64 words aliases every smem
//                 bank).  P7-51 (this revision) instead splits each operand tile into
//                 `NAT = BVK/32` independent **128 B swizzle atoms** (each a separate TMA
//                 box of inner width 32), so BVK==64 keeps the bank-conflict-free swizzle
//                 while still doing the best (BVK=64) tiling -- the two wins §52.6/§53.5
//                 said could not coexist.
#ifdef B2M_TMA
#include <cuda.h>
#define B2M_TMA_ON 1
#else
#define B2M_TMA_ON 0
#endif
// P7-73: `-DB2M_BF16` gives the base B2 K2-mma GEMM core a **bf16-native operand entry**:
// the five streamed operands `do/vn/dv/h/dh` are staged into smem as bf16 (halving both the
// global DRAM read and the smem footprint); the mma stays `m16n8k8.tf32`.  bf16 is a strict
// subset of tf32, so widening bf16->fp32 (exact) before the `cvt.rna.tf32` truncation
// reproduces bf16 (`m16n8k16.bf16`) products exactly -- only the fp32 accumulation order
// differs.  This closes the deployment boundary pinned in P7-64 (§67.4): the CUDA B2 core was
// fp32/tf32-only, so the assembled chain had to `.float()` B1's bf16 `dh`/`dv2` before feeding
// it.  This round covers only the non-TMA `cp.async` staging path and the base GEMM core.
#ifndef B2M_BF16
#define B2M_BF16 0
#endif
// P7-74: `-DB2M_BF16MMA` replaces the GEMM core's tf32 `mma.sync.m16n8k8` with a **native bf16
// `mma.sync.m16n8k16`** (requires `-DB2M_BF16`).  P7-73 staged bf16 operands but kept the tf32
// mma -- valid because bf16 is a strict subset of tf32, so the products stay exact and only the
// fp32 accumulation order differs.  The remaining tf32 limitation (§47.6) is that `.tf32`
// fragments are 32-bit, so there is **no 16-bit `ldmatrix`** and the operands must be read
// straight from smem.  A native bf16 mma halves the mma instruction count (k16 vs k8) and admits
// `.b16` operand feeding; this round isolates the ISA change (same staged bytes, same fragment
// values -- the two-sided packing below yields the exact same operands as the k8 path, only the
// k-grouping / instruction differs).  The GEMM core's direct smem reads are already
// bank-conflict-free by the `OPS=BVK+8` row stride (see docs/07 §77), so `ldmatrix` is *not*
// needed here -- the win looked for is the mma-issue count / tensor density.
#ifndef B2M_BF16MMA
#define B2M_BF16MMA 0
#endif
#if B2M_BF16MMA && !B2M_BF16
#error "B2M_BF16MMA requires B2M_BF16 (bf16 staging)"
#endif
// P7-80: `-DB2M_BF16_VDW` extends the bf16-native operand entry to the two **scratch / raw**
// operands `vv` (raw `v`) and `dwm` (`dw`), which P7-75 kept fp32 in the non-TMA `cp.async` path
// (only the TMA path already staged them bf16, §79).  Under this flag the non-TMA layout matches
// the TMA one: `vv`/`dwm` are `op_t`-strided (`OPS`) bf16 tiles, `prefetch` moves them at the bf16
// `cp.async` granularity (U=8) and `rds()` widens bf16 -> fp32 exactly as `rd()`.  The flag is a
// pure *layout / staging dtype* knob: on bf16-valued `v`/`dw` it is bit-identical to the fp32
// staging (bf16->fp32 widening is exact); the open question it answers is the **precision cost**
// when `v`/`dw` carry full-precision values (docs/07 §83).  It is a no-op under `-DB2M_TMA`
// (already bf16) and requires `-DB2M_BF16`.
#ifndef B2M_BF16_VDW
#define B2M_BF16_VDW 0
#endif
#if B2M_BF16_VDW && !B2M_BF16
#error "B2M_BF16_VDW requires B2M_BF16 (bf16 staging)"
#endif
// P7-86 (diagnostic only): extra bf16 padding (in elements) on the raw-`v` staging tile row stride
// under `-DB2M_BF16_VDW`.  Default 0 = the normal `OPS = BVK + 8`.  It exists solely to test the
// section 88.5 hypothesis that the `V=256` bank-conflict uptick of the `vdw` build comes from the
// bf16 `vv` row-stride phase; it must stay 0 in every deployment build.
#ifndef B2M_VVSTRIDE
#define B2M_VVSTRIDE 0
#endif
#if B2M_VVSTRIDE && (!B2M_BF16_VDW || B2M_TMA_ON)
#error "B2M_VVSTRIDE is a non-TMA BF16_VDW diagnostic knob"
#endif
// P7-60: `-DB2M_PROD` compiles the *production* combination of all six non-GEMM segments in a
// single kernel/launch (see the note further down for the rationale).  It forces the six
// segment flags on, and must be defined before the per-segment `#ifndef` defaults below.
#ifndef B2M_PROD
#define B2M_PROD 0
#endif
#if B2M_PROD
#undef  B2M_OUT
#undef  B2M_DKGB
#undef  B2M_DA
#undef  B2M_SAND
#undef  B2M_DB
#undef  B2M_GATE
#define B2M_OUT  1
#define B2M_DKGB 1
#define B2M_DA   1
#define B2M_SAND 1
#define B2M_DB   1
#define B2M_GATE 1
#endif
// P7-65: cumulative **segment ablation** for attribution (`-DB2M_PROD -DB2M_LEVEL=k`, k in 0..6).
// Unlike the isolated builds (each a separate launcher with its own kernel signature), this keeps
// *every* flag / launch signature / staging path intact and only removes the segment **bodies**
// in reverse dependency order, so the marginal cost `T(k) - T(k-1)` is exactly what segment `k`
// adds *on top of the production kernel*.  Add order: 1=OUT 2=DKGB 3=DA 4=SAND 5=DB 6=GATE
// (INTRA rides with GATE; it needs the full production set).  Level 6 == the P7-60 production
// kernel.  Levels < 5 leave `acc2/acc3` unwritten by design -- this build is for **timing only**.
#if B2M_PROD
#  ifndef B2M_LEVEL
#    define B2M_LEVEL 6
#  endif
#  define B2M_OFF_OUT   (B2M_LEVEL < 1)
#  define B2M_OFF_DKGB  (B2M_LEVEL < 2)
#  define B2M_OFF_DA    (B2M_LEVEL < 3)
#  define B2M_OFF_SAND  (B2M_LEVEL < 4)
#  define B2M_OFF_DB    (B2M_LEVEL < 5)
#  define B2M_OFF_GATE  (B2M_LEVEL < 6)
#  define B2M_OFF_INTRA (B2M_OFF_GATE)
#else
#  define B2M_OFF_OUT   0
#  define B2M_OFF_DKGB  0
#  define B2M_OFF_DA    0
#  define B2M_OFF_SAND  0
#  define B2M_OFF_DB    0
#  define B2M_OFF_GATE  0
#  define B2M_OFF_INTRA 0
#endif
// P7-52: the *first non-GEMM segment* of B2 in CUDA -- the output branch
// (`i_k==0`):   dvb[o_A,v] = sum_{o_t} A[o_A,o_t] * dv[o_t,v]   (== A @ dv)
//               dv2[o_A,v] = beta[o_A] * dvb[o_A,v]             (the complete B2 output `dv2`)
// `dv2` is a *complete* B2 output (never touched by the gate chain afterwards, see
// check_bwd_dqkg_kernel.py:294/297), so it is verifiable standalone.  `dvb` is stored too so
// the A@dv GEMM can be checked directly.  The operand roles differ from the GEMM core: here
// the *contraction* is over chunk rows `o_t` and the *output row* is also a chunk row
// `o_A`, so A is read from `Akk` with a plain index remap (no smem transpose needed) and the
// already-staged `dv` tile serves as the B operand.  See docs/07 §55.
#ifndef B2M_OUT
#define B2M_OUT 0
#endif
// P7-53: the *second non-GEMM segment* of B2 in CUDA -- the `i_k==0` finalize term
//               dkgb[o_A, k] = sum_{o_t} A[o_A, o_t] * (-dw[o_t, k])   (== A @ (-dw))
// (FLA / check_bwd_dqkg_kernel.py:305-307 negate `dw` then `b_dkgb = A @ b_dw`).  Like the
// output branch (P7-52) the *contraction* is over chunk rows `o_t` and the output row is a
// chunk row `o_A`, so `Akk` feeds the mma A-operand by plain index remap (no transpose) and
// the freshly-staged `dw` tile is the B-operand (negated on the fly).  `dkgb` feeds `db`,
// `dg`, `dk` and `dA` downstream, so it is exposed as its own output to check the GEMM
// standalone (mirrors P7-52 exposing `dvb`).
#ifndef B2M_DKGB
#define B2M_DKGB 0
#endif
// P7-61: `-DB2M_HDW` -- keep the GEMM core's `dw` **in-kernel** and feed the DKGB segment
// directly (same-CTA handoff) instead of (a) re-storing the core's own `dw` to global and
// (b) re-staging it back from global as DKGB's B-operand.  Requires `B2M_DKGB`.  This is the
// §49.7 / §63.6 "同 CTA 交接" item: `dw` never leaves the CTA.  Correctness is **bit-identical**
// because the handed-off value IS the core's `accw` -- the very value the global path stored
// and read back (the harness always feeds `db`/`gate`/`prod` with `core.dw`).  The DKGB mma
// therefore moves *post-loop* (the `dw` k-columns are only complete once all V-tiles are
// accumulated), reading a per-warp `swall` mirror of `accw`.  `ov_out`/`dw` staging/dw output
// store are all skipped, so the dw round-trip (~134 MB on `B8T2048H8`) disappears.
#ifndef B2M_HDW
#define B2M_HDW 0
#endif
#if B2M_HDW && !B2M_DKGB
#error "B2M_HDW requires B2M_DKGB"
#endif
// P7-54: the *third non-GEMM segment* of B2 in CUDA -- the two GEMM contributions to `dA`
// (the per-chunk within-chunk attention-matrix adjoint) *before* the final
// `-A^T diag(beta) A^T` sandwich (which is the remaining step):
//   dApre[o_t,o_u] = sum_v dv[o_t,v]*v[o_u,v]                 (base, FLA `dA += dv @ v^T`)
//                  - sum_k dw[o_t,k]*kg[o_u,k]                 (FLA `b_dw = -b_dw; dA += b_dw @ kg^T`)
// with `kg = k * exp2(g)` and `dw` the GEMM core's `dw = dv @ h` output.  NOTE the sign:
// FLA negates `dw` (`chunk_bwd.py:280-281`), so the `dw@kg^T` term enters `dApre` *negated*.  Both contractions
// live *here* (no cross-chunk coupling), so this segment is verifiable standalone like P7-52's
// `dvb` / P7-53's `dkgb`.  The base contracts over V (warp `w` owns V columns [w*16,w*16+16),
// exactly the warp-per-V-column split of the output branch), the `dw@kg^T` contracts over k
// (warp `w` owns output k columns, same split as the GEMM core); each warp produces a private
// `[BT,BT]` partial, the 8 partials are summed through smem.
#ifndef B2M_DA
#define B2M_DA 0
#endif
// P7-66 (this revision): `-DB2M_DAMMA` -- re-express the P7-54 `dApre` segment (both contractions)
// with `mma.sync.m16n8k8.tf32` instead of the scalar FMA loops.  §68.4 pinned DA's `184 us`
// (30% of the bare core) to **scalar shared-memory math** -- it doubles the smem bank conflict
// (9.4M -> 20.8M) while leaving `tensor%` at 1.6% -- and §68.5 named "mma-ize DA" the *only*
// remaining magnitude lever.  Both terms are `X @ Y^T` with the contraction over the full V (base)
// or K (`dw@kg^T`):
//   base :  dApre[o_t,o_u] = sum_{v in warp V-slice} dv[o_t,v] * v[o_u,v]
//   dwkg :  dApre[o_t,o_u]-= sum_{k in warp k-slice} dw[o_t,k] * kg[o_u,k]
// Each warp owns a BW=16-wide contraction slice and computes the *whole* [BT,BT] partial into an
// mma accumulator (2 n-tiles) with 2 mma k-steps per term, then the 8 partials are summed through
// `dapart` exactly as before.  Arithmetic is tf32 (like OUT/DKGB/GEMM), so the standalone check
// moves from the fp32 scalar floor (~5e-7) to the tf32 floor (~3.6e-4).  Requires `B2M_DA`.
#ifndef B2M_DAMMA
#define B2M_DAMMA 0
#endif
#if B2M_DAMMA && !B2M_DA
#error "B2M_DAMMA requires B2M_DA"
#endif
// P7-55: the *fourth non-GEMM segment* of B2 in CUDA -- the final `dA` sandwich (the
// remaining `dA` post-op §55.6/§56.6/§57.6 left open), i.e. the *complete* B2 `dA` output:
//   A where m_A=(o_t>o_u) & valid:   dAm[o_t,o_u] = dApre[o_t,o_u] * beta[o_u]
//   dA = -A^T diag(beta) A^T (dApre)  ==  where(m_A, -(b_A @ (dAm @ b_A)), 0)
// where `b_A[p,j] = A[o_t=j, o_A=p]` (FLA/Triton `b_A` orientation, `check_bwd_dqkg_kernel.py:240`)
// and `dApre` is P7-54's pre-sandwich `dv@v^T - dw@kg^T`.  This consumes `dApre` from the
// in-kernel reduction (no re-read) and only adds two [BT,BT]@[BT,BT] fp32 matmuls + a `beta`
// scale, so it is verifiable standalone against a torch sandwich on the *same* `dApre`
// (bit-level, fp32 scalar) and against FLA's final `dA`.  Requires `B2M_DA`.
#ifndef B2M_SAND
#define B2M_SAND 0
#endif
#if B2M_SAND && !B2M_DA
#error "B2M_SAND requires B2M_DA"
#endif
// P7-63: fold the B3 `intra` diagonal (the last K2-shape structural item, docs/07 §65.6 /
// P7-62's "下一步") into the **production launch** `-DB2M_PROD`.  At deployment CHUNK=16 (NC=1)
// B3 collapses to the diagonal Parts B/D (docs/07 §27.1); `b3_intra_probe.cu` (P7-62) verified
// that diagonal as a K2-shape CUDA kernel.  Here, using the *in-kernel* sandwiched `dAkk`
// (= the masked `-daB` that SAND already produced, so no `dA` round-trip) and the `dAqk` input,
// the same four `[BT,BT]@[BT,KD]` tf32 mmas produce additive `dq/dk/dg` (merged into the gate's
// final outputs, which share the exact mma-fragment layout) and `db` (merged into the `db`
// reduction).  Supersedes the separate B3 kernel: B2+B3 become one K2-shape launch.
// Requires the full production set (SAND's `daB`, GATE, DB, DKGB, DA) -- i.e. `B2M_PROD`.
// P7-67: `B2M_INTRA` is now composable with `B2M_TMA` as well.  The production launch's TMA
// branch originally omitted the trailing `dAqk` kernel argument (the two macros could not be
// built together, docs/07 §67.4 boundary); it is now forwarded just like the non-TMA branch.
#ifndef B2M_INTRA
#define B2M_INTRA 0
#endif
#if B2M_INTRA && !B2M_PROD
#error "B2M_INTRA requires B2M_PROD"
#endif
// P7-56: the *fifth non-GEMM segment* of B2 in CUDA -- the per-token write-gate gradient
//   db[o_A] = sum_v dvb[o_A,v] * v[o_A,v]  +  sum_k dkgb[o_A,k] * kg[o_A,k]
// (FLA `chunk_bwd.py`, `check_bwd_dqkg_kernel.py:180-210`: `b_db = sum_V(b_dvb*b_v)` then
// `b_db += sum_k(b_dkgb*b_kg)`).  Both operands are already computed in the same kernel -- `dvb`
// is the P7-52 output branch (`=A@dv`) and `dkgb` is the P7-53 finalize term (`=A@(-dw)`), each
// owned by a per-warp column slice -- so `db` is exactly a **cross-warp reduction** of two
// warp-private column sums.  It is a *complete* B2 output (never touched by the gate chain), so it
// is verifiable standalone.  Requires the output and dkgb branches (`B2M_OUT` + `B2M_DKGB`), plus
// the raw `v` staging and the chunk `kg=k*exp2(g)` buffer (shared with `B2M_DA`).
#ifndef B2M_DB
#define B2M_DB 0
#endif
#if B2M_DB && !(B2M_OUT && B2M_DKGB)
#error "B2M_DB requires B2M_OUT and B2M_DKGB"
#endif
//   B2M_NODB : keep the `B2M_DB` build/signature/smem but skip the `db` reduction+store
//              (baseline for the isolated `db` segment timing).
#ifndef B2M_NODB
#define B2M_NODB 0
#endif
//   B2M_NODG : keep the `B2M_GATE` build/signature but skip the gate reduction+stores
//              (baseline for the isolated gate-segment timing).
#ifndef B2M_NODG
#define B2M_NODG 0
#endif
// P7-57: the *sixth non-GEMM segment* of B2 in CUDA -- the **gate chain**, i.e. the final
// `dq/dk/dg` from the raw GEMM outputs (`accq`/`acck`), the raw `dgk=[sum_V h*dh]` (already
// computed per chunk) and `dkgb=A@(-dw)` (P7-53).  FLA `chunk_bwd.py` /
// `check_bwd_dqkg_kernel.py:173-189` (`gk_exp=exp2(g)`, `gn=g[last]`, `gb=gk_exp*beta`):
//   dq = dq_raw * gk_exp * scale
//   dk = dk_raw * exp2(gn-g) + dkgb*gb
//   kdk = k * (dk_raw*exp2(gn-g))             (dk *before* the dkgb*gb term)
//   dgk_tot[k] = dgk_raw[k]*exp2(gn[k]) + sum_t kdk[t,k]
//   dg = q*dq - kdk + (t==last)*dgk_tot + kg*dkgb*beta
// Every fragment position of a warp's owned k-column slice is independent; the only reduction
// is `sum_t kdk` over the chunk's rows (an intra-warp shfl over the mma groupID).  Requires
// `B2M_DKGB` (for `dkgb`) and the raw `q`/`k`/`g`/`beta`/`scale` inputs.  Outputs the final
// `dq`,`dk`,`dg` (and keeps `dw`); the raw `dq`/`dk` are *not* stored under `B2M_GATE`
// (production form -- they live on in registers; the harness reads them from the bare core build).
#ifndef B2M_GATE
#define B2M_GATE 0
#endif
#if B2M_GATE && !B2M_DKGB
#error "B2M_GATE requires B2M_DKGB"
#endif
// P7-60: `-DB2M_PROD` allows the *production* combination of all six non-GEMM segments
// (OUT + DKGB + DA + SAND + DB + GATE) in **one** kernel/launch, instead of the
// segment-per-launcher builds the earlier rounds used for isolated verification.  The segment
// code bodies are untouched (their arithmetic, smem buffers and accumulators never overlap --
// each segment owns its own outputs), so the only shared-code changes are (a) allowing the flag
// combination and (b) declaring the shared `beta` operand exactly once.  `B2M_PROD` itself is
// defined/enabled up top (it forces the six flags); this note only records the intent.
// P7-58: coalesce the gate's raw operands `q`/`k`/`g` into smem once per chunk (like `kg`),
// instead of the fragment-address global reads of P7-57 (§60.4: 9.2x global-load sectors).
// The gate then reads `q`/`k`/`g` from the shared buffers (fully coalesced staging load, one
// scalar smem read per fragment position).  Requires `B2M_GATE`.
#ifndef B2M_QKG
#define B2M_QKG 0
#endif
#if B2M_QKG && !B2M_GATE
#error "B2M_QKG requires B2M_GATE"
#endif
//   B2M_QKGP : pad the staged `q`/`k`/`g` row stride from `KD` (128 words = 0 mod 32 banks)
//              to `KD+4`, so the gate's per-fragment shared reads (`row = g/g+8`, fixed col)
//              spread over distinct banks instead of aliasing 8 rows onto one bank.
#ifndef B2M_QKGP
#define B2M_QKGP 0
#endif
#if B2M_QKGP && !B2M_QKG
#error "B2M_QKGP requires B2M_QKG"
#endif
constexpr int QKGP = KD + (B2M_QKGP ? 4 : 0);   // staged q/k/g row stride (words)
// raw `v` staging (`Stage::vv`) and the `kg` buffer are needed by `B2M_DA` (base dA) and/or `B2M_DB`.
#define B2M_VV_KG_ON (B2M_DA || B2M_DB)
constexpr int BVK = B2M_BVK;   // V-tile (contraction) depth staged per pipeline stage
constexpr int NS  = B2M_NS;    // pipeline stages
#if B2M_TMA_ON
// Each operand is staged as NAT atom-major `[NAT][rows][TAW]` sub-buffers, each written by
// its own 128 B-swizzled TMA box.  fp32: 32 elements = 128 B (P7-49/§52).  bf16 (P7-76): an
// atom is 64 elements = 128 B, so NAT=BVK/64 and BVK must be a multiple of 64 (the deployment
// default BVK=32 is not a valid bf16 atom count).  The swizzle decode is byte-identical in both
// (`b ^= ((b>>7)&7)<<4`), because the 128 B atom row stride shifts `row` into bit 7 either way.
// P7-72: BVK==128 -> NAT=4 atoms.  §54.5 left "BVK=128 (4 atom) 未测" open; the loop over
// `a<NAT` and the atom-major Stage layout are NAT-generic, so BVK=128 needs no code
// change -- only a larger dynamic smem, which forces NSTAGE=1 (one 155.6 KB stage > half of
// the 227 KB opt-in budget).  Kept as a *coverage/attribution* point, not a deploy win.
#if B2M_BF16
using tma_t = __nv_bfloat16;
constexpr int TAW = 64;        // atom inner width in elements (64*2 == 128 B)
constexpr int NAT = BVK / 64;
static_assert(BVK == 64 || BVK == 128,
              "bf16 TMA supports BVK in {64,128} (64-element == 128 B atoms)");
#else
using tma_t = float;
constexpr int TAW = 32;        // atom inner width in elements (32*4 == 128 B)
constexpr int NAT = BVK / 32;
static_assert(BVK == 32 || BVK == 64 || BVK == 128,
              "TMA supports BVK in {32,64,128} (128 B atoms)");
#endif
#else
#endif
// padded row stride for the *non-TMA-staged* scratch buffers (skg/swall/dapart/da16), used by
// P7-54 regardless of the stage layout.
constexpr int BVP = BVK + 4;   // padded row stride (16B keeps 4-float cp.async aligned)

// P7-73: bf16-native operand entry (see the macro block at the top).  P7-75 lifts the P7-73
// restriction (base GEMM core only) so `-DB2M_BF16` also covers the full non-TMA production
// launch (`-DB2M_PROD`, all six non-GEMM segments).  The six segments read the five bf16-streamed
// operands (`do/vn/dv/h/dh`) through `rd()` (now bf16-aware) and the **fp32** scratch operands
// (`vv` raw v, `dwm`) through `rds()` (a padded/atom reader that never de-quants bf16); `prefetch`
// stages the fp32 scratch at the fp32 cp.async granularity (U4=4) independently of `op_t`.
// P7-76: `-DB2M_BF16` now also covers the **TMA staging path** (`-DB2M_TMA`).  A bf16 TMA box is
// 64 elements (128 B) wide, so `NAT=BVK/64` (BVK in {64,128}); the swizzle decode is unchanged
// (see the atom-layout note above).  This removes the last `#error` of the bf16 entry.  Under
// BF16+TMA the fp32 scratch operands (`vv`/`dwm`) are also staged bf16 (raw `v` is already bf16
// on the wire; `dw` is a GEMM intermediate the DKGB mma consumes as tf32 anyway), so a single
// bf16 tensor map per operand suffices.  P7-77 (this revision) composes the native-bf16 mma
// (`B2M_BF16MMA`) with TMA as well: `rd2` is generalized to the same `TAW`-atom swizzle decode as
// `rd` (both bf16 of a k16 pair live in the same 128 B atom / 16 B swizzle unit since `col` is
// even), so the combination is no longer excluded.
#if B2M_BF16
using op_t = __nv_bfloat16;
constexpr int OPS = BVK + 8;   // padded bf16 row stride (16 B alignment = 8 bf16)
#define OP_ZERO __float2bfloat16(0.0f)
#else
using op_t = float;
constexpr int OPS = BVP;
#define OP_ZERO 0.0f
#endif

static_assert(VD % BVK == 0, "BVK must divide V");
// P7-81: only the contraction dim `V` is scanned (K != V).  `K` stays 128 because the 16-wide
// warp lane layouts are baked into the `dgk` column reduction (`l & 15`) and the gate chain.
static_assert(KD == 128, "P7-81: only V is parameterized; K=128 is baked into the 16-wide warp reductions");
// P7-82: the six non-GEMM segments are now generic in K != V as well (the V-axis ones use
// `BVW = VD/NW`, the K-axis ones keep `BW = KD/NW`).  The single remaining K/V coupling is the
// **global-`dw` DKGB staging** (`-DB2M_DKGB` without `-DB2M_HDW`): its `dwm` operand is K-tiled
// but rides the V-tile loop `it` (nvt = VD/BVK), so it only addresses the right K range when
// K == V.  The same-CTA handoff (`-DB2M_HDW`) reads the in-kernel `swall`, which is K-axis and
// BW-partitioned, so it is K!=V-clean -- require it for the non-GEMM build when K != V.
#if B2M_DKGB && !B2M_HDW && (B2M_VD != B2M_KD)
#error "P7-82: non-GEMM B2 segments with K!=V require -DB2M_HDW (the global `dw` DKGB staging is K-tiled via the V loop)"
#endif
#if B2M_BF16MMA
static_assert(BVK % 16 == 0, "B2M_BF16MMA contracts 16 V-columns per mma");
#endif

struct alignas(128) Stage {
#if B2M_TMA_ON
    // P7-76: `tma_t`/`TAW` = bf16/64 under `-DB2M_BF16`, else fp32/32.  Each atom is a 128 B
    // SWIZZLE_128B sub-buffer `[rows][TAW]`.
    tma_t do_[NAT][BT][TAW];   // do  [t, v], atom-major (each atom = 128 B-swizzled [BT][TAW])
    tma_t vn [NAT][BT][TAW];   // v_new[t, v]
    tma_t dv [NAT][BT][TAW];   // dv  [t, v]
    tma_t h  [NAT][KD][TAW];   // h  [k, v]  (rows = output N index, cols = contraction v)
    tma_t dh [NAT][KD][TAW];   // dh [k, v]
#if B2M_VV_KG_ON
    tma_t vv [NAT][BT][TAW];   // raw v [t, v]  (base dA / `db` operand; bf16 under BF16+TMA)
#endif
#if B2M_DKGB && !B2M_HDW
    tma_t dwm[NAT][BT][TAW];   // dw [t, k]  (P7-53: B-operand of `A @ -dw`, k-tiled)
#endif
#else
    // P7-73: op_t/OPS are `float`/BVP by default, `__nv_bfloat16`/BVK+8 under -DB2M_BF16.
    op_t do_[BT][OPS];   // do  [t, v]
    op_t vn [BT][OPS];   // v_new[t, v]
    op_t dv [BT][OPS];   // dv  [t, v]
    op_t h  [KD][OPS];   // h  [k, v]  (rows = output N index, cols = contraction v)
    op_t dh [KD][OPS];   // dh [k, v]
#if B2M_VV_KG_ON
#if B2M_BF16_VDW
    op_t  vv [BT][OPS + B2M_VVSTRIDE];   // raw v [t, v]  (bf16 under BF16+VDW)
#else
    float vv [BT][BVP];   // raw v [t, v]
#endif
#endif
#if B2M_DKGB && !B2M_HDW
#if B2M_BF16_VDW
    op_t  dwm[BT][OPS];   // dw [t, k]  (bf16 under BF16+VDW)
#else
    float dwm[BT][BVP];   // dw [t, k]
#endif
#endif
#endif
};
struct alignas(128) Shared {
    Stage st[NS];
#if B2M_TMA_ON
    uint64_t full[NS];    // per-stage TMA-completion mbarrier (arrive count 1)
#endif
#if B2M_VV_KG_ON
    // per-chunk `kg = k*exp2(g)` [BT, KD] (loaded once per chunk, padded row stride), shared
    // by the `B2M_DA` base dA and the `B2M_DB` `sum_k dkgb*kg` term.
    float skg[BT][KD];
#endif
#if B2M_DA || B2M_HDW
    // the per-warp `dw` mirror `swall[NW][BT][BW]` (shared by P7-54 `dw@kg^T` and the P7-61
    // same-CTA handoff feeding DKGB).
    float swall[NW][BT][BW];
#endif
#if B2M_DA
    // the 8 per-warp `dApre` partials `dapart` and the reduced tile `da16`.
    float dapart[NW][BT][BT];
    float da16[BT][BT];
#endif
#if B2M_DB
    // `db` cross-warp reduction: per-warp per-row partials (`dbrow[NW][BT]`) plus the raw-v
    // mirror of each warp's owned V columns (`dbv[NW][BT][BVW]`), since the staged `vv` tile is
    // overwritten before the output branch's `acc2` is complete.
    float dbrow[NW][BT];
    float dbv[NW][BT][BVW];   // P7-82: V-axis mirror -> per-warp V columns (BVW wide)
#endif
#if B2M_SAND
    // b_A[p][j] = A[o_t=j, o_A=p]  (P7-55 sandwich operand; loaded once per chunk)
    float sA[BT][BT];
    float sbeta[BT];     // beta[o_t] (rows t<T)
    float daB[BT][BT];   // scratch: masked dApre, then the second matmul result
#endif
#if B2M_GATE
    // P7-57: per-chunk `dgk_tot[k] = dgk_raw[k]*exp2(g_last[k]) + sum_t kdk[t,k]`, written by
    // the `g==0` lanes of each warp (its own k-column slice) and read back for the `m_last` row.
    float sdgkt[KD];
#endif
#if B2M_QKG
    // P7-58: coalesced per-chunk `q`/`k`/`g` [BT, QKGP] (padded rows; the gate's operand reads
    // become shared loads instead of the P7-57 fragment-address global reads).
    float sq[BT][QKGP];
    float skb[BT][QKGP];
    float sgb[BT][QKGP];
#endif
#if B2M_INTRA
    // P7-63: the folded B3 `intra` diagonal.  Staged raw `q/k/g` [BT,KD] (rows t<T zeroed) and
    // the `dAqk` input tile [BT,BT]; `dAkk` is read directly from SAND's `daB` (masked `-X2`),
    // so no `dA` staging/round-trip.  `idbrow` is the `db` cross-warp reduction (merged into
    // the `db` store).
    float siq[BT][KD];
    float sik[BT][KD];
    float sig[BT][KD];
    float sda[BT][BT];
    float idbrow[NW][BT];
#endif
};

// logical element read from a stage tile.  Non-TMA: padded row-major.  TMA: the operand is
// `[NAT][rows][32]` atom-major; pick the atom by `col>>5`, then apply the SWIZZLE_128B
// decode (Swizzle<3,4,3>: byte bits [4:6] ^= bits [7:9]) *within that atom*.  Since the
// atom stride (`rows*128` bytes) is a multiple of 8 rows, the decode reduces to the same
// `row & 7` XOR as the single-atom BVK==32 case.
DEV float rd(const void* buf, int row, int col, int rows) {
#if B2M_TMA_ON
    // P7-76: generalized to `TAW` (32 fp32 / 64 bf16) elements per 128 B atom; the swizzle
    // decode is identical because the 128 B atom row stride shifts `row` into bit 7 either way.
    const int atom = col / TAW, cc = col % TAW;
    int b = ((atom * rows + row) * TAW + cc) * (int)sizeof(tma_t);
    b = b ^ (((b >> 7) & 7) << 4);
#if B2M_BF16
    return __bfloat162float(*(const __nv_bfloat16*)((const char*)buf + b));
#else
    return *(const float*)((const char*)buf + b);
#endif
#else
    (void)rows;
#if B2M_BF16
    return __bfloat162float(((const __nv_bfloat16*)buf)[row * OPS + col]);
#else
    return ((const float*)buf)[row * BVP + col];
#endif
#endif
}

// P7-54: padded (never TMA-swizzled) row-major read for the `skg`/scratch buffers.
DEV float rdp(const float* buf, int row, int col) { return buf[row * BVP + col]; }
// P7-75: read an **fp32** scratch operand (`vv` raw v / `dwm`) that shares neither the bf16
// de-quantization of `rd` (which would misread fp32 as bf16 under `-DB2M_BF16`) nor, under TMA,
// the atom layout.  Non-TMA: padded fp32 (BVP stride).  TMA: the usual atom-swizzled decode.
// P7-80: under `-DB2M_BF16_VDW` the non-TMA scratch tiles are also bf16 (`op_t`/`OPS`), so the
// read collapses to the same bf16 widening as `rd`.
DEV float rds(const void* buf, int row, int col, int rows) {
#if B2M_TMA_ON
    return rd(buf, row, col, rows);
#elif B2M_BF16_VDW
    (void)rows;
    return __bfloat162float(((const __nv_bfloat16*)buf)[row * OPS + col]);
#else
    (void)rows;
    return ((const float*)buf)[row * BVP + col];
#endif
}
// P7-86 (diagnostic): reader for the raw-`v` staging tile, honoring the `B2M_VVSTRIDE` extra
// padding.  Identical to `rds` at the default `B2M_VVSTRIDE == 0`.
DEV float rdsvv(const void* buf, int row, int col) {
#if B2M_TMA_ON
    return rd(buf, row, col, BT);
#elif B2M_BF16_VDW
    return __bfloat162float(((const __nv_bfloat16*)buf)[row * (OPS + B2M_VVSTRIDE) + col]);
#else
    return ((const float*)buf)[row * BVP + col];
#endif
}

#if B2M_TMA_ON
// ---- mbarrier / TMA (same primitives as b1_tma_probe.cu) --------------------
DEV uint32_t saddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint64_t* b, uint32_t cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(saddr(b)), "r"(cnt));
}
DEV void mbar_arrive_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(saddr(b)), "r"(bytes) : "memory");
}
DEV void mbar_wait(uint64_t* b, int ph) {
    asm volatile(
        "{\n\t.reg .pred P;\n"
        "LAB_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@!P bra LAB_%=;\n}\n" ::"r"(saddr(b)), "r"(ph) : "memory");
}
DEV void tma4d_load(const CUtensorMap* m, void* sm, int c0, int c1, int c2, int c3,
                    uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2,%3,%4,%5}], [%6];\n"
        ::"r"(saddr(sm)), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(saddr(bar))
        : "memory");
}
DEV void tma5d_load(const CUtensorMap* m, void* sm, int c0, int c1, int c2, int c3, int c4,
                    uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];\n"
        ::"r"(saddr(sm)), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
          "r"(saddr(bar))
        : "memory");
}
#endif

DEV uint32_t to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}
// D = A*B + C, m16n8k8, A row-major, B col-major, tf32 x tf32 -> fp32
DEV void mma8(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
#if B2M_BF16MMA
// P7-74: pack the two consecutive bf16 at `buf[row*OPS + col .. col+1]` into one 32-bit register
// (little-endian: low half = col, high half = col+1).  This is exactly the per-lane payload of
// the bf16 m16n8k16 A/B fragments.
// P7-77: under `-DB2M_TMA` the operand lives in the same `[NAT][rows][TAW]` 128 B-swizzle atom
// layout `rd` decodes; `col` is even (k16 fragment pairs 2t/2t+8), so `col` and `col+1` fall in
// the same atom and the same 16 B swizzle unit -> decode `col`'s byte address once, read 2 bf16.
DEV uint32_t rd2(const void* buf, int row, int col, int rows) {
#if B2M_TMA_ON
    const int atom = col / TAW, cc = col % TAW;
    int b = ((atom * rows + row) * TAW + cc) * (int)sizeof(tma_t);
    b = b ^ (((b >> 7) & 7) << 4);
    const unsigned short lo =
        __bfloat16_as_ushort(*(const __nv_bfloat16*)((const char*)buf + b));
    const unsigned short hi =
        __bfloat16_as_ushort(*(const __nv_bfloat16*)((const char*)buf + b + (int)sizeof(tma_t)));
#else
    (void)rows;
    const unsigned short lo = __bfloat16_as_ushort(((const __nv_bfloat16*)buf)[row * OPS + col]);
    const unsigned short hi = __bfloat16_as_ushort(((const __nv_bfloat16*)buf)[row * OPS + col + 1]);
#endif
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
// D = A*B + C, m16n8k16, A row-major, B col-major, bf16 x bf16 -> fp32.
// A fragment (lane l, g=l>>2, t=l&3):  a0={A[g][2t],A[g][2t+1]}, a1={A[g+8][2t],A[g+8][2t+1]},
//                                      a2={A[g][2t+8],A[g][2t+9]}, a3={A[g+8][2t+8],A[g+8][2t+9]}.
// B fragment:                          b0={B[2t][g],B[2t+1][g]}, b1={B[2t+8][g],B[2t+9][g]}.
// C fragment: identical to m16n8k8 (c0/c1 row g, cols 2t/2t+1; c2/c3 row g+8).
DEV void mma16(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
#endif

// ---------------------------------------------------------------- stage prefetch
#if !B2M_TMA_ON
DEV void prefetch(Stage* st, const float* __restrict__ v, const float* __restrict__ vn,
                  const float* __restrict__ h, const float* __restrict__ dh,
                  const float* __restrict__ do_, const float* __restrict__ dv,
#if B2M_DKGB
                  const float* __restrict__ dw,
#endif
                  int n, int hh, int T, int H, int NT, int c, int v0) {
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int t0 = c * BT;
#if B2M_BF16
    constexpr int U = 8;   // bf16 per 16-byte cp.async
    const __nv_bfloat16* gdo = (const __nv_bfloat16*)do_;
    const __nv_bfloat16* gvn = (const __nv_bfloat16*)vn;
    const __nv_bfloat16* gdv = (const __nv_bfloat16*)dv;
    const __nv_bfloat16* gh  = (const __nv_bfloat16*)h;
    const __nv_bfloat16* gdh = (const __nv_bfloat16*)dh;
#else
    constexpr int U = 4;   // fp32 per 16-byte cp.async
    const float* gdo = do_; const float* gvn = vn; const float* gdv = dv;
    const float* gh = h; const float* gdh = dh;
#endif
    // do/vn/dv [BT, BVK]  (token rows; zero the pad rows for a non-divisible tail)
    for (int idx = tid; idx < BT * (BVK / U); idx += nthr) {
        int row = idx / (BVK / U), cg = idx % (BVK / U), t = t0 + row;
        op_t* pd = &st->do_[row][cg * U];
        op_t* pv = &st->vn [row][cg * U];
        op_t* pw = &st->dv [row][cg * U];
#if B2M_NODOV
        for (int j = 0; j < U; ++j) { pd[j] = OP_ZERO; pv[j] = OP_ZERO; pw[j] = OP_ZERO; }
#else
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + hh) * VD + v0 + cg * U;
            cp16(pd, gdo + off);
            cp16(pv, gvn + off);
            cp16(pw, gdv + off);
        } else {
            for (int j = 0; j < U; ++j) { pd[j] = OP_ZERO; pv[j] = OP_ZERO; pw[j] = OP_ZERO; }
        }
#endif
    }
    // P7-75: the **fp32** scratch operands (`vv` raw v for base dA / `db`, and `dwm` for the
    // `A@-dw` segment) are staged at the fp32 cp.async granularity (U4=4), independent of the bf16
    // `op_t` operands above (whose U is 8 under `-DB2M_BF16`).
    // P7-80: under `-DB2M_BF16_VDW` they are staged as bf16 (U4=8 on the bf16 wire), matching the
    // TMA path's `tma_t` layout; the element index arithmetic is unchanged.
#if B2M_VV_KG_ON || (B2M_DKGB && !B2M_HDW)
    {
#if B2M_BF16_VDW
        constexpr int U4 = 8;   // bf16 per 16-byte cp.async
        const __nv_bfloat16* gv = (const __nv_bfloat16*)v;
  #if B2M_DKGB && !B2M_HDW
        const __nv_bfloat16* gdw = (const __nv_bfloat16*)dw;
  #endif
        using vdw_t = __nv_bfloat16;
#else
        constexpr int U4 = 4;   // fp32 per 16-byte cp.async
        const float* gv = v;
  #if B2M_DKGB && !B2M_HDW
        const float* gdw = dw;
  #endif
        using vdw_t = float;
#endif
        for (int idx = tid; idx < BT * (BVK / U4); idx += nthr) {
            int row = idx / (BVK / U4), cg = idx % (BVK / U4), t = t0 + row;
#if B2M_VV_KG_ON
            vdw_t* pvv = &st->vv[row][cg * U4];
  #if B2M_NODOV
            for (int j = 0; j < U4; ++j) pvv[j] = (vdw_t)0;
  #else
            if (t < T) {
                const int64_t off = ((int64_t)(n * T + t) * H + hh) * VD + v0 + cg * U4;
                cp16(pvv, gv + off);
            } else {
                for (int j = 0; j < U4; ++j) pvv[j] = (vdw_t)0;
            }
  #endif
#endif
#if B2M_DKGB && !B2M_HDW
            // dw [BT, BVK] (token rows; k-tiled).  Same [B,T,H,K] layout as do/vn/dv, KD==VD.
            vdw_t* pn = &st->dwm[row][cg * U4];
            if (t < T) {
                const int64_t off = ((int64_t)(n * T + t) * H + hh) * KD + v0 + cg * U4;
                cp16(pn, gdw + off);
            } else {
                for (int j = 0; j < U4; ++j) pn[j] = (vdw_t)0;
            }
#endif
        }
    }
#endif
    // h/dh [KD, BVK]  (chunk-granular, always full)
    const int64_t hbase = ((int64_t)(n * NT + c) * H + hh) * KD * VD;
    for (int idx = tid; idx < KD * (BVK / U); idx += nthr) {
        int row = idx / (BVK / U), cg = idx % (BVK / U);
        const int64_t off = hbase + (int64_t)row * VD + v0 + cg * U;
#if B2M_NOHDH
        for (int j = 0; j < U; ++j) { st->h[row][cg * U + j] = OP_ZERO; st->dh[row][cg * U + j] = OP_ZERO; }
#else
        cp16(&st->h [row][cg * U], gh  + off);
        cp16(&st->dh[row][cg * U], gdh + off);
#endif
    }
}
#endif  // !B2M_TMA_ON

#if B2M_TMA_ON
// Single-thread TMA issue for one V-tile stage: do/vn/dv [BT,BVK] + h/dh [KD,BVK].  OOB
// token rows (non-divisible T) are zero-filled by TMA automatically.  `bar` is the stage's
// `full` mbarrier; `expect_tx` is the *full box* byte count (TMA counts zero-filled bytes).
// Each operand is issued as `NAT` 128 B (32 fp32) inner-width boxes into its own atom.
DEV void issue_tma(Stage* st, const CUtensorMap* dom, const CUtensorMap* vnm,
                   const CUtensorMap* dvm, const CUtensorMap* hm, const CUtensorMap* dhm,
#if B2M_DKGB
                   const CUtensorMap* dwm,
#endif
#if B2M_VV_KG_ON
                   const CUtensorMap* vvm,
#endif
                   int n, int hh, int c, int v0, uint64_t* bar) {
    const int t0 = c * BT;
    mbar_arrive_tx(bar, (uint32_t)(((3
#if B2M_VV_KG_ON
                                    + 1
#endif
#if B2M_DKGB && !B2M_HDW
                                    + 1
#endif
                                   ) * BT + 2 * KD) * BVK * (int)sizeof(tma_t)));
    #pragma unroll
    for (int a = 0; a < NAT; ++a) {
        const int va = v0 + a * TAW;
        tma4d_load(dom, &st->do_[a][0][0], va, t0, hh, n, bar);
        tma4d_load(vnm, &st->vn [a][0][0], va, t0, hh, n, bar);
        tma4d_load(dvm, &st->dv [a][0][0], va, t0, hh, n, bar);
#if B2M_VV_KG_ON
        tma4d_load(vvm, &st->vv [a][0][0], va, t0, hh, n, bar);
#endif
        tma5d_load(hm,  &st->h  [a][0][0], va, 0, hh, c, n, bar);
        tma5d_load(dhm, &st->dh [a][0][0], va, 0, hh, c, n, bar);
#if B2M_DKGB && !B2M_HDW
        tma4d_load(dwm, &st->dwm[a][0][0], va, t0, hh, n, bar);
#endif
    }
}
#endif

// ---------------------------------------------------------------------- kernel
#if B2M_DKGB
#define TMA_DWM_ARG , &dwm
#define PRE_DW_ARG  , dw
#else
#define TMA_DWM_ARG
#define PRE_DW_ARG
#endif
#if B2M_VV_KG_ON
#define TMA_VV_ARG , &vvm
#else
#define TMA_VV_ARG
#endif
#if B2M_GATE
// trailing kernel args of the standalone gate build (after `dkgb`): beta + q/k/g + gated outs
#define GATE_TAIL , (const float*)beta, (const float*)qin, (const float*)kin, \
                   (const float*)gin, (float*)dqg, (float*)dkg, (float*)dg, scale
#else
#define GATE_TAIL
#endif
__global__ void __launch_bounds__(NW * 32)
b2mma_kernel(
#if B2M_TMA_ON
             const __grid_constant__ CUtensorMap dom, const __grid_constant__ CUtensorMap vnm,
             const __grid_constant__ CUtensorMap dvm, const __grid_constant__ CUtensorMap hm,
             const __grid_constant__ CUtensorMap dhm,
#if B2M_DKGB
             const __grid_constant__ CUtensorMap dwm,
#endif
#if B2M_VV_KG_ON
             const __grid_constant__ CUtensorMap vvm,
#endif
#endif
             const float* __restrict__ v, const float* __restrict__ vn,
             const float* __restrict__ h, const float* __restrict__ dh,
             const float* __restrict__ do_, const float* __restrict__ dv,
#if B2M_VV_KG_ON
             const float* __restrict__ kptr, const float* __restrict__ gptr,
#endif
#if B2M_DKGB
             const float* __restrict__ dw,
#endif
             float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dw_out,
             float* __restrict__ dgk,
#if B2M_OUT || B2M_DKGB
             const float* __restrict__ A4,
#endif
#if B2M_OUT
             const float* __restrict__ beta,
             float* __restrict__ dv2, float* __restrict__ dvb,
#endif
#if B2M_DKGB
             float* __restrict__ dkgb,
#endif
#if B2M_GATE && !B2M_OUT
             const float* __restrict__ beta,
#endif
#if B2M_GATE
             const float* __restrict__ qptr, const float* __restrict__ kraw,
             const float* __restrict__ graw,
             float* __restrict__ dqg, float* __restrict__ dkg, float* __restrict__ dg,
             float scale,
#endif
#if B2M_DB
             float* __restrict__ db_out,
#endif
#if B2M_DA
             float* __restrict__ dApre,
#endif
#if B2M_SAND
             const float* __restrict__ A4s,
#if !B2M_OUT && !B2M_GATE
             const float* __restrict__ beta,
#endif
             float* __restrict__ dAout,
#endif
#if B2M_INTRA
             const float* __restrict__ dAqk,
#endif
             int B, int T, int H, int NT) {
    extern __shared__ char smem_raw[];
    Shared* sh = reinterpret_cast<Shared*>(smem_raw);

    const int i_t = blockIdx.x, i_bh = blockIdx.y;
    const int n = i_bh / H, hh = i_bh % H;
    const int l = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = l >> 2, t = l & 3;   // groupID, threadID-in-group

    float accq[2][4], acck[2][4], accw[2][4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int x = 0; x < 4; ++x) { accq[nt][x] = 0.f; acck[nt][x] = 0.f; accw[nt][x] = 0.f; }
    float gpart = 0.f;
    float gpart1 = 0.f;   // B2M_DGKCF second kcol half (kcol = warp*BW + 8 + c)
#if B2M_INTRA
    // P7-63: folded B3 `intra` diagonal contributions (same mma-fragment layout as the gate's
    // final `dq/dk/dg`), produced after SAND and merged into the GATE/DB stores below.
    float idq[2][4], idk[2][4], idg[2][4];
#endif
#if B2M_OUT
    float acc2[NBV][4];   // dvb[o_A, v] output tile (m16 x NBV*n8), warp-private V columns
    #pragma unroll
    for (int nt = 0; nt < NBV; ++nt)
        #pragma unroll
        for (int x = 0; x < 4; ++x) acc2[nt][x] = 0.f;
#endif
#if B2M_DKGB
    float acc3[2][4];     // dkgb[o_A, k] output tile (m16 x 2*n8), warp-private k columns
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int x = 0; x < 4; ++x) acc3[nt][x] = 0.f;
#endif
#if B2M_DA && !B2M_DAMMA
    float dbpart[8], dkpart[8];   // per-lane partials: 8 [r,c] entries each (r=l>>1, c=(l&1)*8+i)
    #pragma unroll
    for (int i = 0; i < 8; ++i) { dbpart[i] = 0.f; dkpart[i] = 0.f; }
#endif
#if B2M_DA && B2M_DAMMA
    // P7-66: mma accumulators -- each warp computes the *full* [BT,BT] partial (2 n-tiles of
    // m16n8) for both terms, over its own BW-wide contraction slice.  Same C layout as accq.
    float dAacc[2][4], dKacc[2][4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int x = 0; x < 4; ++x) { dAacc[nt][x] = 0.f; dKacc[nt][x] = 0.f; }
#endif
#if B2M_VV_KG_ON
    // `kg = k*exp2(g)` for this chunk (rows t<T; pad rows zeroed).  One load per chunk (not per
    // V-tile); read back post-loop by `rdp` (P7-54 `dw@kg^T`) and by the P7-56 `db` term.
    {
        for (int idx = threadIdx.x; idx < BT * KD; idx += NW * 32) {
            const int row = idx / KD, col = idx % KD, tt = i_t * BT + row;
            float kv = 0.f, gv = 0.f;
            if (tt < T) {
                const int64_t off = ((int64_t)(n * T + tt) * H + hh) * KD + col;
                kv = kptr[off]; gv = gptr[off];
            }
            sh->skg[row][col] = kv * exp2f(gv);
        }
    }
#endif
#if B2M_SAND
    // b_A[p][j] = A[b, t=i_t*BT+j, h, c=p]  (only rows t<T valid); beta[o_t] (rows t<T).
    for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
        const int j = idx / BT, p = idx % BT;
        float av = 0.f;
        if (i_t * BT + j < T)
            av = A4s[((int64_t)(n * T + i_t * BT + j) * H + hh) * BT + p];
        sh->sA[p][j] = av;
    }
    if (threadIdx.x < BT) {
        const int r = threadIdx.x;
        sh->sbeta[r] = (i_t * BT + r < T) ? beta[((int64_t)(n * T + i_t * BT + r) * H + hh)] : 0.f;
    }
#endif
#if B2M_QKG
    // P7-58: coalesced per-chunk `q`/`k`/`g` (rows t<T; pad rows zeroed).  One pass per chunk;
    // the gate reads these from smem instead of the fragment-address global reads (P7-57 §60.4).
    for (int idx = threadIdx.x; idx < BT * KD; idx += NW * 32) {
        const int row = idx / KD, col = idx % KD, tt = i_t * BT + row;
        float qv = 0.f, kv = 0.f, gv = 0.f;
        if (tt < T) {
            const int64_t off = ((int64_t)(n * T + tt) * H + hh) * KD + col;
            qv = qptr[off]; kv = kraw[off]; gv = graw[off];
        }
        sh->sq[row][col] = qv; sh->skb[row][col] = kv; sh->sgb[row][col] = gv;
    }
#endif
#if B2M_DA || B2M_VV_KG_ON || B2M_SAND || B2M_QKG
    __syncthreads();
#endif
#if B2M_OUT || B2M_DKGB
    const int64_t a_base = ((int64_t)(n * T + i_t * BT) * H + hh) * (int64_t)BT;
#endif

    const int nvt = VD / BVK;
#if B2M_TMA_ON
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < NS; ++s) mbar_init(&sh->full[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int s = 0; s < NS - 1; ++s)
            if (s < nvt)
                issue_tma(&sh->st[s], &dom, &vnm, &dvm, &hm, &dhm TMA_DWM_ARG TMA_VV_ARG, n, hh, i_t, s * BVK,
                          &sh->full[s]);
    }
    for (int it = 0; it < nvt; ++it) {
        if (threadIdx.x == 0) {
            const int snext = it + NS - 1;
            if (snext < nvt)
                issue_tma(&sh->st[snext % NS], &dom, &vnm, &dvm, &hm, &dhm TMA_DWM_ARG TMA_VV_ARG, n, hh, i_t,
                          snext * BVK, &sh->full[snext % NS]);
        }
        mbar_wait(&sh->full[it % NS], (it / NS) & 1);
        __syncthreads();

        const Stage* st = &sh->st[it % NS];
#else
    for (int s = 0; s < NS - 1; ++s) {
        if (s < nvt) prefetch(&sh->st[s], v, vn, h, dh, do_, dv PRE_DW_ARG, n, hh, T, H, NT, i_t, s * BVK);
        __pipeline_commit();
    }

    for (int it = 0; it < nvt; ++it) {
        const int snext = it + NS - 1;
        if (snext < nvt)
            prefetch(&sh->st[snext % NS], v, vn, h, dh, do_, dv PRE_DW_ARG, n, hh, T, H, NT, i_t, snext * BVK);
        __pipeline_commit();
        __pipeline_wait_prior(NS - 1);
        __syncthreads();

        const Stage* st = &sh->st[it % NS];
#endif
#if B2M_BF16MMA
        // P7-74: native bf16 `mma.m16n8k16` (k16 per instruction; same staged bytes/fragments).
        const __nv_bfloat16* pdo = (const __nv_bfloat16*)st->do_;
        const __nv_bfloat16* pvn = (const __nv_bfloat16*)st->vn;
        const __nv_bfloat16* pdv = (const __nv_bfloat16*)st->dv;
        const __nv_bfloat16* ph  = (const __nv_bfloat16*)st->h;
        const __nv_bfloat16* pdh = (const __nv_bfloat16*)st->dh;
        #pragma unroll
        for (int kk = 0; kk < BVK / 16; ++kk) {
            const int kc = kk * 16;
            // A fragments (m16 x k16) from do/vn/dv:
            //   a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t],..} a2={A[g][2t+8],..} a3={A[g+8][2t+8],..}
            uint32_t aq[4], ak[4], aw[4];
            aq[0] = rd2(pdo, g    , kc + 2 * t, BT);
            aq[1] = rd2(pdo, g + 8, kc + 2 * t, BT);
            aq[2] = rd2(pdo, g    , kc + 2 * t + 8, BT);
            aq[3] = rd2(pdo, g + 8, kc + 2 * t + 8, BT);
            ak[0] = rd2(pvn, g    , kc + 2 * t, BT);
            ak[1] = rd2(pvn, g + 8, kc + 2 * t, BT);
            ak[2] = rd2(pvn, g    , kc + 2 * t + 8, BT);
            ak[3] = rd2(pvn, g + 8, kc + 2 * t + 8, BT);
            aw[0] = rd2(pdv, g    , kc + 2 * t, BT);
            aw[1] = rd2(pdv, g + 8, kc + 2 * t, BT);
            aw[2] = rd2(pdv, g    , kc + 2 * t + 8, BT);
            aw[3] = rd2(pdv, g + 8, kc + 2 * t + 8, BT);
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int krow = warp * BW + nt * 8 + g;   // output column = k = n index
                // B (col-major k16 x n8): b0={B[2t][g],B[2t+1][g]}, b1={B[2t+8][g],B[2t+9][g]};
                // with h[n][k] the B[k][n]=h[n][k] element is h[krow][kc+2t..].
                uint32_t bh[2], bd[2];
                bh[0] = rd2(ph , krow, kc + 2 * t, KD);
                bh[1] = rd2(ph , krow, kc + 2 * t + 8, KD);
                bd[0] = rd2(pdh, krow, kc + 2 * t, KD);
                bd[1] = rd2(pdh, krow, kc + 2 * t + 8, KD);
                mma16(accq[nt], aq, bh, accq[nt]);
                mma16(acck[nt], ak, bd, acck[nt]);
                mma16(accw[nt], aw, bh, accw[nt]);
            }
        }
#else
        #pragma unroll
        for (int kk = 0; kk < BVK / 8; ++kk) {
            const int kc = kk * 8;
            // A fragments (m16 x k8) from do/vn/dv: A[row=g/g+8][col=t/t+4]
            uint32_t aq[4], ak[4], aw[4];
            aq[0] = to_tf32(rd((const float*)st->do_, g    , kc + t, BT));
            aq[1] = to_tf32(rd((const float*)st->do_, g + 8, kc + t, BT));
            aq[2] = to_tf32(rd((const float*)st->do_, g    , kc + t + 4, BT));
            aq[3] = to_tf32(rd((const float*)st->do_, g + 8, kc + t + 4, BT));
            ak[0] = to_tf32(rd((const float*)st->vn, g    , kc + t, BT));
            ak[1] = to_tf32(rd((const float*)st->vn, g + 8, kc + t, BT));
            ak[2] = to_tf32(rd((const float*)st->vn, g    , kc + t + 4, BT));
            ak[3] = to_tf32(rd((const float*)st->vn, g + 8, kc + t + 4, BT));
            aw[0] = to_tf32(rd((const float*)st->dv, g    , kc + t, BT));
            aw[1] = to_tf32(rd((const float*)st->dv, g + 8, kc + t, BT));
            aw[2] = to_tf32(rd((const float*)st->dv, g    , kc + t + 4, BT));
            aw[3] = to_tf32(rd((const float*)st->dv, g + 8, kc + t + 4, BT));
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int krow = warp * BW + nt * 8 + g;   // output column = k = n index
                uint32_t bh[2], bd[2];
                bh[0] = to_tf32(rd((const float*)st->h , krow, kc + t, KD));
                bh[1] = to_tf32(rd((const float*)st->h , krow, kc + t + 4, KD));
                bd[0] = to_tf32(rd((const float*)st->dh, krow, kc + t, KD));
                bd[1] = to_tf32(rd((const float*)st->dh, krow, kc + t + 4, KD));
                mma8(accq[nt], aq, bh, accq[nt]);
                mma8(acck[nt], ak, bd, acck[nt]);
                mma8(accw[nt], aw, bh, accw[nt]);
            }
        }
#endif
#if !B2M_NODGK
  #if B2M_DGKCF
        // Conflict-free `dgk` column reduction.  smem word addr = kcol*BVP + vv with
        // BVP=36 -> bank = (4*kcol + vv) & 31; 16 consecutive kcols therefore alias as
        // (kcol, kcol+8).  Use only 8 kcols per instruction: c = l&7 (kcol low bits),
        // s = l>>3 (the vv subclass), so banks = 4*c + s cover 0..31 exactly.
        {
            const int c = l & 7, s = l >> 3;
            #pragma unroll
            for (int ph = 0; ph < 2; ++ph) {
                const int kcol = warp * BW + ph * 8 + c;
                float p = 0.f;
                for (int vv = s; vv < BVK; vv += 4)
                    p += rd((const float*)st->h, kcol, vv, KD) * rd((const float*)st->dh, kcol, vv, KD);
                if (ph == 0) gpart += p; else gpart1 += p;
            }
        }
  #else
        // baseline dgk column reduction: lane l handles column warp*BW + (l&15), split v by l>>4
        {
            const int kcol = warp * BW + (l & 15);
            float p = 0.f;
            for (int vv = (l >> 4); vv < BVK; vv += 2)
                p += rd((const float*)st->h, kcol, vv, KD) * rd((const float*)st->dh, kcol, vv, KD);
            gpart += p;
        }
  #endif
#endif
#if B2M_OUT && !B2M_OFF_OUT
        // ---- output branch (P7-52): dvb = A @ dv, contraction over chunk rows o_t ----
        // Warp `w` owns V columns [w*BVW, w*BVW+BVW) (P7-82: V-axis warp width); only the stage
        // whose V-tile contains them contributes.  A[o_A,o_t] = Akk[token=i_t*BT+o_t, o_A]
        // (index remap, no transpose); the B operand is the already-staged `dv` tile.
        {
            const int local0 = warp * BVW - it * BVK;
            if (local0 >= 0 && local0 + BVW <= BVK) {
                #pragma unroll
                for (int kk = 0; kk < BT / 8; ++kk) {
                    const int kc = kk * 8;
                    const int ot0 = kc + t, ot1 = kc + t + 4;
                    const float a00 = (i_t * BT + ot0) < T ? A4[a_base + (int64_t)ot0 * (H * BT) + g]     : 0.f;
                    const float a10 = (i_t * BT + ot0) < T ? A4[a_base + (int64_t)ot0 * (H * BT) + g + 8] : 0.f;
                    const float a01 = (i_t * BT + ot1) < T ? A4[a_base + (int64_t)ot1 * (H * BT) + g]     : 0.f;
                    const float a11 = (i_t * BT + ot1) < T ? A4[a_base + (int64_t)ot1 * (H * BT) + g + 8] : 0.f;
                    const uint32_t af[4] = {to_tf32(a00), to_tf32(a10), to_tf32(a01), to_tf32(a11)};
                    #pragma unroll
                    for (int nt = 0; nt < NBV; ++nt) {
                        const int lc = local0 + nt * 8 + g;
                        uint32_t bf[2];
                        bf[0] = to_tf32(rd((const float*)st->dv, ot0, lc, BT));
                        bf[1] = to_tf32(rd((const float*)st->dv, ot1, lc, BT));
                        mma8(acc2[nt], af, bf, acc2[nt]);
                    }
                }
            }
        }
#endif
#if B2M_DKGB && !B2M_HDW && !B2M_OFF_DKGB
        // ---- second non-GEMM segment (P7-53): dkgb = A @ (-dw), contraction over o_t ----
        // Structurally identical to the output branch above (A-operand index remap, no
        // transpose), except the B-operand is the freshly-staged `dw` tile (k-tiled, warp
        // `w` owns k columns [w*BW,w*BW+BW)) and its sign is flipped (`b_dw = -b_dw`, FLA
        // chunk_bwd.py:305 / check_bwd_dqkg_kernel.py:305).
        {
            const int local0 = warp * BW - it * BVK;
            if (local0 >= 0 && local0 + BW <= BVK) {
                #pragma unroll
                for (int kk = 0; kk < BT / 8; ++kk) {
                    const int kc = kk * 8;
                    const int ot0 = kc + t, ot1 = kc + t + 4;
                    const float a00 = (i_t * BT + ot0) < T ? A4[a_base + (int64_t)ot0 * (H * BT) + g]     : 0.f;
                    const float a10 = (i_t * BT + ot0) < T ? A4[a_base + (int64_t)ot0 * (H * BT) + g + 8] : 0.f;
                    const float a01 = (i_t * BT + ot1) < T ? A4[a_base + (int64_t)ot1 * (H * BT) + g]     : 0.f;
                    const float a11 = (i_t * BT + ot1) < T ? A4[a_base + (int64_t)ot1 * (H * BT) + g + 8] : 0.f;
                    const uint32_t af[4] = {to_tf32(a00), to_tf32(a10), to_tf32(a01), to_tf32(a11)};
                    #pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int lc = local0 + nt * 8 + g;
                        uint32_t bf[2];
                        bf[0] = to_tf32(-rds(st->dwm, ot0, lc, BT));
                        bf[1] = to_tf32(-rds(st->dwm, ot1, lc, BT));
                        mma8(acc3[nt], af, bf, acc3[nt]);
                    }
                }
            }
        }
#endif
#if B2M_DA && !B2M_DAMMA && !B2M_OFF_DA
        // ---- P7-54 base dA: dApre[o_t,o_u] += sum_{v in warp V cols} dv[o_t,v]*v[o_u,v] ----
        // Warp `w` owns V columns [w*BVW, w*BVW+BVW) (P7-82 V-axis width, only matching stage).
        // Lane `l` owns the 8 entries (r = l>>1, c = (l&1)*8 + i, i=0..7) of the [BT,BT] partial.
        {
            const int local0 = warp * BVW - it * BVK;
            if (local0 >= 0 && local0 + BVW <= BVK) {
                const int r = l >> 1, c0 = (l & 1) * 8;
                #pragma unroll
                for (int j = 0; j < BVW; ++j) {
                    const float dvr = rd((const float*)st->dv, r, local0 + j, BT);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                        dbpart[i] += dvr * rdsvv(st->vv, c0 + i, local0 + j);
                }
            }
        }
#endif
#if B2M_DA && B2M_DAMMA && !B2M_OFF_DA
        // ---- P7-66 base dA via mma: dApre[o_t,o_u] += dv @ v^T over the warp's V slice ----
        // A = dv[o_t, v], B = v[o_u, v] (row.col -> B is read as v[o_u,k]); contraction = warp's
        // BVW V columns (P7-82 V-axis width; BVW/8 mma k-steps of 8); the whole [BT,BT] partial
        // lives in 2 n-tiles (output axes are o_t/o_u, not V).
        {
            const int local0 = warp * BVW - it * BVK;
            if (local0 >= 0 && local0 + BVW <= BVK) {
                #pragma unroll
                for (int kk = 0; kk < BVW / 8; ++kk) {
                    const int kc = local0 + kk * 8;
                    const uint32_t a[4] = {
                        to_tf32(rd((const float*)st->dv, g,     kc + t,     BT)),
                        to_tf32(rd((const float*)st->dv, g + 8, kc + t,     BT)),
                        to_tf32(rd((const float*)st->dv, g,     kc + t + 4, BT)),
                        to_tf32(rd((const float*)st->dv, g + 8, kc + t + 4, BT))};
                    #pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        uint32_t b[2];
                        b[0] = to_tf32(rdsvv(st->vv, nt * 8 + g, kc + t));
                        b[1] = to_tf32(rdsvv(st->vv, nt * 8 + g, kc + t + 4));
                        mma8(dAacc[nt], a, b, dAacc[nt]);
                    }
                }
            }
        }
#endif
#if B2M_DB && !B2M_NODB && !B2M_OFF_DB
        // ---- P7-56: mirror the raw `v` of this warp's owned V columns into `dbv` (the staged
        // `vv` tile is overwritten before `acc2` is complete, so `db`'s `dvb*v` term needs a copy).
        {
            const int local0 = warp * BVW - it * BVK;
            if (local0 >= 0 && local0 + BVW <= BVK) {
                for (int idx = l; idx < BT * BVW; idx += 32) {
                    const int r = idx / BVW, j = idx % BVW;
                    sh->dbv[warp][r][j] = rdsvv(st->vv, r, local0 + j);
                }
            }
        }
#endif
        __syncthreads();
    }

    // ---- store ----
    const int t0 = i_t * BT;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col = warp * BW + nt * 8 + 2 * t;
        const int tr0 = t0 + g, tr1 = t0 + g + 8;
        const int64_t o0 = ((int64_t)(n * T + tr0) * H + hh) * KD + col;
        const int64_t o1 = ((int64_t)(n * T + tr1) * H + hh) * KD + col;
        if (tr0 < T) {
#if B2M_VECST
  #if !B2M_GATE
            *reinterpret_cast<float2*>(&dq[o0]) = make_float2(accq[nt][0], accq[nt][1]);
            *reinterpret_cast<float2*>(&dk[o0]) = make_float2(acck[nt][0], acck[nt][1]);
  #endif
  #if !B2M_HDW
            *reinterpret_cast<float2*>(&dw_out[o0]) = make_float2(accw[nt][0], accw[nt][1]);
  #endif
#else
  #if !B2M_GATE
            dq[o0] = accq[nt][0]; dq[o0 + 1] = accq[nt][1];
            dk[o0] = acck[nt][0]; dk[o0 + 1] = acck[nt][1];
  #endif
  #if !B2M_HDW
            dw_out[o0] = accw[nt][0]; dw_out[o0 + 1] = accw[nt][1];
  #endif
#endif
        }
        if (tr1 < T) {
#if B2M_VECST
  #if !B2M_GATE
            *reinterpret_cast<float2*>(&dq[o1]) = make_float2(accq[nt][2], accq[nt][3]);
            *reinterpret_cast<float2*>(&dk[o1]) = make_float2(acck[nt][2], acck[nt][3]);
  #endif
  #if !B2M_HDW
            *reinterpret_cast<float2*>(&dw_out[o1]) = make_float2(accw[nt][2], accw[nt][3]);
  #endif
#else
  #if !B2M_GATE
            dq[o1] = accq[nt][2]; dq[o1 + 1] = accq[nt][3];
            dk[o1] = acck[nt][2]; dk[o1 + 1] = acck[nt][3];
  #endif
  #if !B2M_HDW
            dw_out[o1] = accw[nt][2]; dw_out[o1 + 1] = accw[nt][3];
  #endif
#endif
        }
    }
  #if B2M_DGKCF
    // reduce over s in {0,1,2,3} = lanes {l, l^8, l^16, l^24}; lane l<8 then holds both
    // halves' sums for kcol = warp*BW + l and warp*BW + 8 + l.
    gpart  += __shfl_xor_sync(0xffffffffu, gpart,  8);
    gpart1 += __shfl_xor_sync(0xffffffffu, gpart1, 8);
    gpart  += __shfl_xor_sync(0xffffffffu, gpart,  16);
    gpart1 += __shfl_xor_sync(0xffffffffu, gpart1, 16);
    if (l < 8) {
        const int64_t base = ((int64_t)(n * NT + i_t) * H + hh) * KD + warp * BW + l;
        dgk[base] = gpart;
        dgk[base + 8] = gpart1;
    }
  #else
    gpart += __shfl_xor_sync(0xffffffffu, gpart, 16);
    if (l < 16) {
        const int kcol = warp * BW + l;
        dgk[((int64_t)(n * NT + i_t) * H + hh) * KD + kcol] = gpart;
    }
  #endif
#if B2M_OUT && !B2M_OFF_OUT
    // ---- store `dv2` (complete B2 output) and `dvb` = A@dv (P7-82: V-axis warp width) ----
    #pragma unroll
    for (int nt = 0; nt < NBV; ++nt) {
        const int col = warp * BVW + nt * 8 + 2 * t;
        const int tr0 = t0 + g, tr1 = t0 + g + 8;
        const int64_t o0 = ((int64_t)(n * T + tr0) * H + hh) * VD + col;
        const int64_t o1 = ((int64_t)(n * T + tr1) * H + hh) * VD + col;
        if (tr0 < T) {
            const float bt0 = beta[((int64_t)(n * T + tr0) * H + hh)];
            st2(&dvb[o0], acc2[nt][0], acc2[nt][1]);
            st2(&dv2[o0], bt0 * acc2[nt][0], bt0 * acc2[nt][1]);
        }
        if (tr1 < T) {
            const float bt1 = beta[((int64_t)(n * T + tr1) * H + hh)];
            st2(&dvb[o1], acc2[nt][2], acc2[nt][3]);
            st2(&dv2[o1], bt1 * acc2[nt][2], bt1 * acc2[nt][3]);
        }
    }
#endif
#if B2M_DKGB && B2M_HDW && !B2M_OFF_DKGB
    // ---- P7-61 same-CTA handoff: run `dkgb = A @ (-dw)` **post-loop** off the in-kernel
    // `dw` (=accw, complete only after all V-tiles) instead of a global round-trip.  Mirror
    // this warp's `dw` fragment into `swall` exactly as the P7-54 `dw@kg^T` path does, then
    // contract over o_t (warp `w` owns k columns [w*BW,w*BW+BW), i.e. its own `swall` slice).
    {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = nt * 8 + 2 * t;
            sh->swall[warp][g][col]         = accw[nt][0];
            sh->swall[warp][g][col + 1]     = accw[nt][1];
            sh->swall[warp][g + 8][col]     = accw[nt][2];
            sh->swall[warp][g + 8][col + 1] = accw[nt][3];
        }
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < BT / 8; ++kk) {
            const int kc = kk * 8;
            const int ot0 = kc + t, ot1 = kc + t + 4;
            const float a00 = (i_t * BT + ot0) < T ? A4[a_base + (int64_t)ot0 * (H * BT) + g]     : 0.f;
            const float a10 = (i_t * BT + ot0) < T ? A4[a_base + (int64_t)ot0 * (H * BT) + g + 8] : 0.f;
            const float a01 = (i_t * BT + ot1) < T ? A4[a_base + (int64_t)ot1 * (H * BT) + g]     : 0.f;
            const float a11 = (i_t * BT + ot1) < T ? A4[a_base + (int64_t)ot1 * (H * BT) + g + 8] : 0.f;
            const uint32_t af[4] = {to_tf32(a00), to_tf32(a10), to_tf32(a01), to_tf32(a11)};
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int lc = nt * 8 + g;   // local k column within this warp's slice
                uint32_t bf[2];
                bf[0] = to_tf32(-sh->swall[warp][ot0][lc]);
                bf[1] = to_tf32(-sh->swall[warp][ot1][lc]);
                mma8(acc3[nt], af, bf, acc3[nt]);
            }
        }
    }
#endif
#if B2M_DKGB && !B2M_OFF_DKGB
    // ---- store `dkgb` = A @ (-dw) (row = o_A token axis, col = output k) ----
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col = warp * BW + nt * 8 + 2 * t;
        const int tr0 = t0 + g, tr1 = t0 + g + 8;
        const int64_t o0 = ((int64_t)(n * T + tr0) * H + hh) * KD + col;
        const int64_t o1 = ((int64_t)(n * T + tr1) * H + hh) * KD + col;
        if (tr0 < T) st2(&dkgb[o0], acc3[nt][0], acc3[nt][1]);
        if (tr1 < T) st2(&dkgb[o1], acc3[nt][2], acc3[nt][3]);
    }
#endif
#if B2M_DA && !B2M_OFF_DA
    // ---- P7-54 post-loop: `dw @ kg^T` partial, reduce the 8 warps, store `dApre` ----
    {
#if !B2M_DAMMA
        const int r = l >> 1, c0 = (l & 1) * 8;
#endif
        // Mirror this warp's `dw` (=accw, mma-fragment layout: rows g/g+8, cols 2t/2t+1) into
        // `swall[warp]` so any lane can read it in row-major order.
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = nt * 8 + 2 * t;
            sh->swall[warp][g][col]         = accw[nt][0];
            sh->swall[warp][g][col + 1]     = accw[nt][1];
            sh->swall[warp][g + 8][col]     = accw[nt][2];
            sh->swall[warp][g + 8][col + 1] = accw[nt][3];
        }
        __syncwarp();
#if B2M_DAMMA
        // P7-66: `dw@kg^T` via mma over this warp's output k columns [warp*BW,warp*BW+BW).
        // A = dw = swall[warp][o_t, k-local], B = kg = skg[o_u, warp*BW + k-local].  FLA negates
        // `dw` (chunk_bwd.py:280-281), so this term is *subtracted* from the base below.
        #pragma unroll
        for (int kk = 0; kk < BW / 8; ++kk) {
            const int kc = kk * 8;
            const uint32_t a[4] = {
                to_tf32(sh->swall[warp][g][kc + t]),     to_tf32(sh->swall[warp][g + 8][kc + t]),
                to_tf32(sh->swall[warp][g][kc + t + 4]), to_tf32(sh->swall[warp][g + 8][kc + t + 4])};
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                uint32_t b[2];
                b[0] = to_tf32(sh->skg[nt * 8 + g][warp * BW + kc + t]);
                b[1] = to_tf32(sh->skg[nt * 8 + g][warp * BW + kc + t + 4]);
                mma8(dKacc[nt], a, b, dKacc[nt]);
            }
        }
        // write the [BT,BT] partial (mma C layout) as dAacc - dKacc
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int c = nt * 8 + 2 * t;
            sh->dapart[warp][g][c]             = dAacc[nt][0] - dKacc[nt][0];
            sh->dapart[warp][g][c + 1]         = dAacc[nt][1] - dKacc[nt][1];
            sh->dapart[warp][g + 8][c]         = dAacc[nt][2] - dKacc[nt][2];
            sh->dapart[warp][g + 8][c + 1]     = dAacc[nt][3] - dKacc[nt][3];
        }
#else
        // dw@kg^T partial over this warp's output k columns [warp*BW, warp*BW+BW).  FLA negates
        // `dw`, so the term is *subtracted* from the base below (matches `chunk_bwd.py:280-281`).
        #pragma unroll
        for (int kk = 0; kk < BW; ++kk) {
            const float w0 = sh->swall[warp][r][kk];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                dkpart[i] += w0 * sh->skg[c0 + i][warp * BW + kk];
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            sh->dapart[warp][r][c0 + i] = dbpart[i] - dkpart[i];
#endif
    }
    __syncthreads();
    {   // reduce the 8 warp partials (one thread per [BT,BT] entry)
        const int idx = threadIdx.x, r = idx / BT, c = idx % BT;
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) s += sh->dapart[w][r][c];
        sh->da16[r][c] = s;
    }
    __syncthreads();
    {   // store `dApre` (row = o_t, col = o_u; rows o_t<T).  P7-70: pair adjacent columns
        // (c, c+1) into one `st2` (`STG.E.64` under `B2M_VECST`, two scalar otherwise); the
        // `BT*BT` thread-per-entry store was already fully coalesced, this only halves the
        // store instruction count.  BT is even so the row splits into `BT/2` column pairs.
        const int idx = threadIdx.x;
        if (idx < BT * (BT / 2)) {
            const int r = idx / (BT / 2), c = 2 * (idx % (BT / 2)), tr = i_t * BT + r;
            if (tr < T)
                st2(&dApre[((int64_t)(n * T + tr) * H + hh) * BT + c],
                    sh->da16[r][c], sh->da16[r][c + 1]);
        }
    }
#endif
#if B2M_SAND && !B2M_OFF_SAND
    // ---- P7-55 final dA sandwich: dA = where(m_A, -(b_A @ ((dApre*beta_col) @ b_A)), 0) ----
    // `sh->da16` still holds `dApre` [o_t,o_u]; all ops are fp32 scalar (no tf32), so the
    // result is bit-comparable to a torch sandwich on the *same* `dApre`.
    __syncthreads();
    {   // dAm[r][c] = (r>c & valid) ? dApre[r][c]*beta[c] : 0
        const int idx = threadIdx.x;
        if (idx < BT * BT) {
            const int r = idx / BT, c = idx % BT;
            const bool v = (r > c) && (i_t * BT + r < T) && (i_t * BT + c < T);
            sh->daB[r][c] = v ? sh->da16[r][c] * sh->sbeta[c] : 0.f;
        }
    }
    __syncthreads();
    {   // X1[i][j] = sum_p dAm[i][p] * b_A[p][j]   -> da16
        for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
            const int i = idx / BT, jj = idx % BT;
            float s = 0.f;
            #pragma unroll
            for (int p = 0; p < BT; ++p) s += sh->daB[i][p] * sh->sA[p][jj];
            sh->da16[i][jj] = s;
        }
    }
    __syncthreads();
    {   // X2[i][j] = sum_p b_A[i][p] * X1[p][j]     -> daB
        for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
            const int i = idx / BT, jj = idx % BT;
            float s = 0.f;
            #pragma unroll
            for (int p = 0; p < BT; ++p) s += sh->sA[i][p] * sh->da16[p][jj];
            sh->daB[i][jj] = s;
        }
    }
    __syncthreads();
    {   // store final `dA` = where(m_A, -X2, 0).  X2 first axis maps to the stored token row
        // (golden/Triton use the same-index mapping, `check_bwd_dqkg_kernel.py:203`), so token
        // row = i and column index = jj, with the mask (i>jj) applied on X2's own axes.
        for (int idx = threadIdx.x; idx < BT * (BT / 2); idx += NW * 32) {
            const int i = idx / (BT / 2), jj = 2 * (idx % (BT / 2));
            const bool v0 = (i > jj) && (i_t * BT + i < T) && (i_t * BT + jj < T);
            const bool v1 = (i > jj + 1) && (i_t * BT + i < T) && (i_t * BT + jj + 1 < T);
            if (i_t * BT + i < T)   // P7-70: pair columns (jj, jj+1) into one `st2`
                st2(&dAout[((int64_t)(n * T + i_t * BT + i) * H + hh) * BT + jj],
                    v0 ? -sh->daB[i][jj] : 0.f, v1 ? -sh->daB[i][jj + 1] : 0.f);
        }
    }
#endif
#if B2M_INTRA && !B2M_OFF_INTRA
    // ---- P7-63: folded B3 `intra` diagonal (docs/07 §65/§66).  `daB` still holds SAND's
    // sandwiched `X2`; `dAkk = (a>j & valid) ? -daB[a][j] : 0`.  Four `[BT,BT]@[BT,KD]` mmas
    // (warp owns output k-columns [w*BW, w*BW+BW)) -> additive `dq/dk/dg/db`.
    for (int idx = threadIdx.x; idx < BT * KD; idx += NW * 32) {
        const int row = idx / KD, col = idx % KD, tt = i_t * BT + row;
        float qv = 0.f, kv = 0.f, gv = 0.f;
        if (tt < T) {
            const int64_t off = ((int64_t)(n * T + tt) * H + hh) * KD + col;
            qv = qptr[off]; kv = kraw[off]; gv = graw[off];
        }
        sh->siq[row][col] = qv; sh->sik[row][col] = kv; sh->sig[row][col] = gv;
    }
    for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
        const int a = idx / BT, j = idx % BT;
        const bool vv = (i_t * BT + a < T) && (i_t * BT + j < T);
        const int64_t off = ((int64_t)(n * T + i_t * BT + a) * H + hh) * BT + j;
        sh->sda[a][j] = vv ? dAqk[off] : 0.f;
    }
    __syncthreads();
    {
        float idt[2][4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) { idq[nt][i] = 0.f; idk[nt][i] = 0.f; idt[nt][i] = 0.f; }
        // in-kernel sandwiched `dAkk` = masked `-daB`
        #define SAKK(a, j) (((a) > (j) && (i_t * BT + (a)) < T && (i_t * BT + (j)) < T) \
                                ? -sh->daB[(a)][(j)] : 0.f)
        #pragma unroll
        for (int kk = 0; kk < BT / 8; ++kk) {
            const int kc = kk * 8;
            const int j0 = kc + t, j1 = kc + t + 4;
            uint32_t mq[4] = {to_tf32(sh->sda[g][j0]),     to_tf32(sh->sda[g + 8][j0]),
                              to_tf32(sh->sda[g][j1]),     to_tf32(sh->sda[g + 8][j1])};
            uint32_t mk[4] = {to_tf32(SAKK(g, j0)),        to_tf32(SAKK(g + 8, j0)),
                              to_tf32(SAKK(g, j1)),        to_tf32(SAKK(g + 8, j1))};
            uint32_t sq_[4] = {to_tf32(sh->sda[j0][g]),    to_tf32(sh->sda[j0][g + 8]),
                               to_tf32(sh->sda[j1][g]),    to_tf32(sh->sda[j1][g + 8])};
            uint32_t sk_[4] = {to_tf32(SAKK(j0, g)),       to_tf32(SAKK(j0, g + 8)),
                               to_tf32(SAKK(j1, g)),       to_tf32(SAKK(j1, g + 8))};
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int col = warp * BW + nt * 8 + g;
                const float kg0 = sh->sik[j0][col] * exp2f(-sh->sig[j0][col]);
                const float kg1 = sh->sik[j1][col] * exp2f(-sh->sig[j1][col]);
                const float qg0 = sh->siq[j0][col] * exp2f(sh->sig[j0][col]);
                const float qg1 = sh->siq[j1][col] * exp2f(sh->sig[j1][col]);
                const float kd0 = sh->sik[j0][col] * sh->sbeta[j0] * exp2f(sh->sig[j0][col]);
                const float kd1 = sh->sik[j1][col] * sh->sbeta[j1] * exp2f(sh->sig[j1][col]);
                uint32_t b_kg[2]  = {to_tf32(kg0), to_tf32(kg1)};
                uint32_t b_qg[2]  = {to_tf32(qg0), to_tf32(qg1)};
                uint32_t b_kgd[2] = {to_tf32(kd0), to_tf32(kd1)};
                mma8(idq[nt], mq, b_kg, idq[nt]);
                mma8(idk[nt], mk, b_kg, idk[nt]);
                mma8(idt[nt], sq_, b_qg, idt[nt]);
                mma8(idt[nt], sk_, b_kgd, idt[nt]);
            }
        }
        // gating (rows g/g+8, cols nt*8+2t+(i&1)); `dkpre` saved for `db`
        float dkpre[2][4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rowrel = (i < 2) ? g : g + 8;
                const int col = warp * BW + nt * 8 + 2 * t + (i & 1);
                const float gg = sh->sig[rowrel][col];
                idq[nt][i] *= exp2f(gg);                 // dq2 = 2^g ⊙ (Mq@KG)
                idk[nt][i] *= exp2f(gg);                 // dk2 = 2^g ⊙ (Mk@KG)
                idt[nt][i] *= exp2f(-gg);                // dkt = 2^-g ⊙ (Sq@QG + Sk@KGd)
                dkpre[nt][i] = idk[nt][i];
                idk[nt][i] *= sh->sbeta[rowrel];          // dk2 *= beta
            }
        }
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rowrel = (i < 2) ? g : g + 8;
                const int col = warp * BW + nt * 8 + 2 * t + (i & 1);
                const float kv = sh->sik[rowrel][col];
                idg[nt][i] = sh->siq[rowrel][col] * idq[nt][i] + (idk[nt][i] - idt[nt][i]) * kv;
                idk[nt][i] += idt[nt][i];                 // dk2 += dkt
            }
        }
        // db partial: sum_k dkpre * k over this warp's k columns, then cross-warp
        float pr0 = 0.f, pr1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rowrel = (i < 2) ? g : g + 8;
                const int col = warp * BW + nt * 8 + 2 * t + (i & 1);
                const float p = dkpre[nt][i] * sh->sik[rowrel][col];
                if (i < 2) pr0 += p; else pr1 += p;
            }
        }
        pr0 += __shfl_xor_sync(0xffffffffu, pr0, 1);
        pr0 += __shfl_xor_sync(0xffffffffu, pr0, 2);
        pr1 += __shfl_xor_sync(0xffffffffu, pr1, 1);
        pr1 += __shfl_xor_sync(0xffffffffu, pr1, 2);
        if (t == 0) { sh->idbrow[warp][g] = pr0; sh->idbrow[warp][g + 8] = pr1; }
        #undef SAKK
    }
#endif
#if B2M_GATE && !B2M_NODG && !B2M_OFF_GATE
    // ---- P7-57: gate chain -> final `dq`/`dk`/`dg` (FLA `chunk_bwd.py:173-189`) ----
    __syncthreads();   // raw `dgk` stores visible block-wide before the read-back below
    {
        const int last = ((i_t + 1) * BT < T ? (i_t + 1) * BT : T) - 1;
        const int lastrow = last - t0;                    // relative row of the last valid token
        const int64_t dgk_base = ((int64_t)(n * NT + i_t) * H + hh) * KD;
#if !B2M_QKG
        const int64_t gl_base = ((int64_t)(n * T + last) * H + hh) * KD;
#endif
        float dgv[2][4];
        float dqgv[2][4], dkgv[2][4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int cl = nt * 8 + 2 * t;                // column within the warp's owned 16
            const int c0 = warp * BW + cl, c1 = c0 + 1;
#if B2M_QKG
            const float gl0 = sh->sgb[lastrow][c0], gl1 = sh->sgb[lastrow][c1];
#else
            const float gl0 = graw[gl_base + c0], gl1 = graw[gl_base + c1];   // g at (last, col)
#endif
            const float f0 = exp2f(gl0), f1 = exp2f(gl1); // exp2(g_last) for dgk_raw
            const int tr0 = t0 + g, tr1 = t0 + g + 8;
            float pk0 = 0.f, pk1 = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = (i < 2) ? tr0 : tr1;
                const int col = (i & 1) ? c1 : c0;
                float qv = 0.f, kv = 0.f, gv = 0.f, bt = 0.f;
                if (row < T) {
#if B2M_QKG
                    const int rowrel = (i < 2) ? g : g + 8;
                    qv = sh->sq[rowrel][col]; kv = sh->skb[rowrel][col]; gv = sh->sgb[rowrel][col];
#else
                    const int64_t off = ((int64_t)(n * T + row) * H + hh) * KD + col;
                    qv = qptr[off]; kv = kraw[off]; gv = graw[off];
#endif
                    bt = beta[((int64_t)(n * T + row) * H + hh)];
                }
                const float gk = exp2f(gv);
                const float gnm = exp2f(((i & 1) ? gl1 : gl0) - gv);   // exp2(g_last - g)
                const float dqf = accq[nt][i] * gk * scale;
                const float dkm = acck[nt][i] * gnm;
                const float dgb = acc3[nt][i];
                const float kdk = kv * dkm;
#if B2M_INTRA
                dgv[nt][i] = qv * dqf - kdk + kv * gk * dgb * bt + idg[nt][i];
                dqgv[nt][i] = dqf + idq[nt][i];
                dkgv[nt][i] = dkm + dgb * gk * bt + idk[nt][i];
#else
                dgv[nt][i] = qv * dqf - kdk + kv * gk * dgb * bt;
                dqgv[nt][i] = dqf;
                dkgv[nt][i] = dkm + dgb * gk * bt;
#endif
                if (i == 0 || i == 2) pk0 += kdk; else pk1 += kdk;
            }
#if B2M_VECST
            // pair (c0,c1) -> one `STG.E.64` per row (row tr0 for i={0,1}, tr1 for i={2,3})
            {
                const int64_t r0 = ((int64_t)(n * T + tr0) * H + hh) * KD + c0;
                const int64_t r1 = ((int64_t)(n * T + tr1) * H + hh) * KD + c0;
                if (tr0 < T) { st2(&dqg[r0], dqgv[nt][0], dqgv[nt][1]);
                               st2(&dkg[r0], dkgv[nt][0], dkgv[nt][1]); }
                if (tr1 < T) { st2(&dqg[r1], dqgv[nt][2], dqgv[nt][3]);
                               st2(&dkg[r1], dkgv[nt][2], dkgv[nt][3]); }
            }
#else
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = (i < 2) ? tr0 : tr1;
                const int col = (i & 1) ? c1 : c0;
                if (row < T) {
                    const int64_t off = ((int64_t)(n * T + row) * H + hh) * KD + col;
                    dqg[off] = dqgv[nt][i];
                    dkg[off] = dkgv[nt][i];
                }
            }
#endif
            // `sum_t kdk` over the chunk: reduce over the mma groupID (lane bits 2..4)
            pk0 += __shfl_xor_sync(0xffffffffu, pk0, 4);
            pk0 += __shfl_xor_sync(0xffffffffu, pk0, 8);
            pk0 += __shfl_xor_sync(0xffffffffu, pk0, 16);
            pk1 += __shfl_xor_sync(0xffffffffu, pk1, 4);
            pk1 += __shfl_xor_sync(0xffffffffu, pk1, 8);
            pk1 += __shfl_xor_sync(0xffffffffu, pk1, 16);
            if (g == 0) {
                sh->sdgkt[c0] = dgk[dgk_base + c0] * f0 + pk0;
                sh->sdgkt[c1] = dgk[dgk_base + c1] * f1 + pk1;
            }
        }
        __syncwarp();
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int cl = nt * 8 + 2 * t;
            const int c0 = warp * BW + cl, c1 = c0 + 1;
            const float dl0 = sh->sdgkt[c0], dl1 = sh->sdgkt[c1];
            const int tr0 = t0 + g, tr1 = t0 + g + 8;
#if B2M_VECST
            const float a00 = (g == lastrow) ? dl0 : 0.f;
            const float a01 = (g == lastrow) ? dl1 : 0.f;
            const float a10 = (g + 8 == lastrow) ? dl0 : 0.f;
            const float a11 = (g + 8 == lastrow) ? dl1 : 0.f;
            const int64_t r0 = ((int64_t)(n * T + tr0) * H + hh) * KD + c0;
            const int64_t r1 = ((int64_t)(n * T + tr1) * H + hh) * KD + c0;
            if (tr0 < T) st2(&dg[r0], dgv[nt][0] + a00, dgv[nt][1] + a01);
            if (tr1 < T) st2(&dg[r1], dgv[nt][2] + a10, dgv[nt][3] + a11);
#else
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int row = (i < 2) ? tr0 : tr1;
                const int col = (i & 1) ? c1 : c0;
                if (row < T) {
                    const int rowrel = (i < 2) ? g : g + 8;
                    const float add = (rowrel == lastrow)
                                          ? ((i & 1) ? dl1 : dl0) : 0.f;
                    dg[((int64_t)(n * T + row) * H + hh) * KD + col] = dgv[nt][i] + add;
                }
            }
#endif
        }
    }
#endif
#if B2M_DB && !B2M_NODB && !B2M_OFF_DB
    // ---- P7-56: db = sum_v dvb*v + sum_k dkgb*kg (cross-warp reduction of column sums) ----
    // acc2/acc3 fragment rows are g/g+8, cols nt*8+2t,nt*8+2t+1 (within the warp's owned slice).
    // `skg` is persistent (ensures the `dkgb*kg` term can be read post-loop); `dbv` mirrors the
    // warp's V columns of raw `v` for the `dvb*v` term.
    {
        float p0 = 0.f, p1 = 0.f;
        // P7-82: the two terms are on different axes -- `acc2` (dvb) is V-axis, `acc3` (dkgb) is
        // K-axis -- so with K != V they have different per-warp widths (`BVW` vs `BW`).  At K==V
        // (NBV==2) the NBV-major loop below reproduces the original interleave acc2/acc3/acc2/acc3
        // exactly, so the reduction is bit-identical to the pre-P7-82 kernel.
        constexpr int NM = NBV > 2 ? NBV : 2;
        #pragma unroll
        for (int nt = 0; nt < NM; ++nt) {
            const int cl = nt * 8 + 2 * t;
            if (nt < NBV) {   // dvb * raw-v : V columns [warp*BVW, +BVW)
                p0 += acc2[nt][0] * sh->dbv[warp][g][cl]     + acc2[nt][1] * sh->dbv[warp][g][cl + 1];
                p1 += acc2[nt][2] * sh->dbv[warp][g + 8][cl] + acc2[nt][3] * sh->dbv[warp][g + 8][cl + 1];
            }
            if (nt < 2) {     // dkgb * kg : K columns [warp*BW, +BW)
                const int ck = warp * BW + cl;
                p0 += acc3[nt][0] * sh->skg[g][ck]           + acc3[nt][1] * sh->skg[g][ck + 1];
                p1 += acc3[nt][2] * sh->skg[g + 8][ck]       + acc3[nt][3] * sh->skg[g + 8][ck + 1];
            }
        }
        p0 += __shfl_xor_sync(0xffffffffu, p0, 1); p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
        p1 += __shfl_xor_sync(0xffffffffu, p1, 1); p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
        if (t == 0) { sh->dbrow[warp][g] = p0; sh->dbrow[warp][g + 8] = p1; }
    }
    __syncthreads();
    {
        const int idx = threadIdx.x;
        if (idx < BT) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) s += sh->dbrow[w][idx];
#if B2M_INTRA && !B2M_OFF_INTRA
            #pragma unroll
            for (int w = 0; w < NW; ++w) s += sh->idbrow[w][idx];
#endif
            if (i_t * BT + idx < T)
                db_out[((int64_t)(n * T + i_t * BT + idx) * H + hh)] = s;
        }
    }
#endif
}

#if B2M_TMA_ON
// do/vn/dv are [B, T, H, V] fp32 contiguous.  dims {V, T, H, B}, box {32, BT, 1, 1},
// SWIZZLE_128B (inner 32*4 == 128 B).  One box per 128 B atom (NAT of them per operand).
// OOB token rows of a non-divisible T zero-fill.
// P7-76: element type / size / atom inner width follow `tma_t` (bf16/2/64 under -DB2M_BF16,
// else fp32/4/32).  `esz` is in bytes so the global strides scale with the datatype.
static constexpr cuuint64_t TMA_ESZ = sizeof(tma_t);
static constexpr cuuint32_t TMA_INW = TAW;
static constexpr CUtensorMapDataType TMA_DT =
    B2M_BF16 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
static void make_map_dov(CUtensorMap* m, void* ptr, int H, int T, int B) {
    cuuint64_t gdim[4] = {(cuuint64_t)VD, (cuuint64_t)T, (cuuint64_t)H, (cuuint64_t)B};
    cuuint64_t gstr[3] = {(cuuint64_t)H * VD * TMA_ESZ, (cuuint64_t)VD * TMA_ESZ,
                          (cuuint64_t)T * H * VD * TMA_ESZ};
    cuuint32_t box[4] = {TMA_INW, (cuuint32_t)BT, 1, 1};
    cuuint32_t estr[4] = {1, 1, 1, 1};
    if (cuTensorMapEncodeTiled(m, TMA_DT, 4, ptr, gdim, gstr, box,
                               estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        fprintf(stderr, "cuTensorMapEncodeTiled(dov) failed\n");
}
// h/dh are [B, NT, H, KD, VD] fp32 contiguous.  dims {VD, KD, H, NT, B},
// box {32, KD, 1, 1, 1}, SWIZZLE_128B (128 B rows; swizzle repeats every 8 rows).
static void make_map_hdh(CUtensorMap* m, void* ptr, int H, int NT, int B) {
    cuuint64_t gdim[5] = {(cuuint64_t)VD, (cuuint64_t)KD, (cuuint64_t)H, (cuuint64_t)NT,
                          (cuuint64_t)B};
    cuuint64_t gstr[4] = {(cuuint64_t)VD * TMA_ESZ, (cuuint64_t)KD * VD * TMA_ESZ,
                          (cuuint64_t)H * KD * VD * TMA_ESZ, (cuuint64_t)NT * H * KD * VD * TMA_ESZ};
    cuuint32_t box[5] = {TMA_INW, (cuuint32_t)KD, 1, 1, 1};
    cuuint32_t estr[5] = {1, 1, 1, 1, 1};
    if (cuTensorMapEncodeTiled(m, TMA_DT, 5, ptr, gdim, gstr, box,
                               estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        fprintf(stderr, "cuTensorMapEncodeTiled(hdh) failed\n");
}
#endif

#if !B2M_DA && !B2M_DB
static int b2mma_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv,
#if B2M_DKGB
                             const void* dwin,
#endif
                             void* dq, void* dk, void* dw, void* dgk,
#if B2M_OUT || B2M_DKGB
                             const void* A4,
#endif
#if B2M_OUT
                             const void* beta, void* dv2, void* dvb,
#endif
#if B2M_DKGB
                             void* dkgb,
#endif
#if B2M_GATE
                             const void* beta, const void* qin, const void* kin,
                             const void* gin, void* dqg, void* dkg, void* dg, float scale,
#endif
                             int B, int T, int H, int NT, void* stream) {
    dim3 grid(NT, B * H);
    int smem = (int)sizeof(Shared);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b2mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
#if B2M_TMA_ON
    CUtensorMap dom, vnm, dvm, hm, dhm;
    make_map_dov(&dom, (void*)dov, H, T, B);
    make_map_dov(&vnm, (void*)vn,  H, T, B);
    make_map_dov(&dvm, (void*)dvv, H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
#if B2M_DKGB
    CUtensorMap dwm;
    make_map_dov(&dwm, (void*)dwin, H, T, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, dwm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        (const float*)A4, (float*)dkgb GATE_TAIL,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
#if B2M_OUT
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
#endif
        B, T, H, NT);
#endif
#else
#if B2M_DKGB
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        (const float*)A4, (float*)dkgb GATE_TAIL,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
#if B2M_OUT
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
#endif
        B, T, H, NT);
#endif
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2mma_launch(const void* v, const void* vn, const void* h, const void* dh,
                            const void* dov, const void* dvv,
                            void* dq, void* dk, void* dw, void* dgk,
                            int B, int T, int H, int NT, void* stream) {
#if B2M_OUT || B2M_DKGB
    // Only meaningful when compiled without -DB2M_OUT/-DB2M_DKGB; those builds must use
    // their own entries (b2out_launch / b2dkgb_launch), else null operands reach the kernel.
    return -4;
#else
    return b2mma_launch_impl(v, vn, h, dh, dov, dvv, dq, dk, dw, dgk, B, T, H, NT, stream);
#endif
}

#if B2M_OUT
extern "C" int b2out_launch(const void* v, const void* vn, const void* h, const void* dh,
                           const void* dov, const void* dvv, const void* A4, const void* beta,
                           void* dq, void* dk, void* dw, void* dgk, void* dv2, void* dvb,
                           int B, int T, int H, int NT, void* stream) {
    return b2mma_launch_impl(v, vn, h, dh, dov, dvv, dq, dk, dw, dgk,
                             A4, beta, dv2, dvb, B, T, H, NT, stream);
}
#endif

#if B2M_DKGB && !B2M_GATE
extern "C" int b2dkgb_launch(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv, const void* dwin,
                             const void* A4,
                             void* dq, void* dk, void* dw, void* dgk, void* dkgb,
                             int B, int T, int H, int NT, void* stream) {
    return b2mma_launch_impl(v, vn, h, dh, dov, dvv, dwin, dq, dk, dw, dgk,
                             A4, dkgb, B, T, H, NT, stream);
}
#endif

#if B2M_GATE
// P7-57 dedicated launcher (`b2gate_launch`): compiled with `-DB2M_DKGB -DB2M_GATE`
// (standalone; no B2M_OUT/B2M_SAND).  Emits the raw `dq/dk/dw/dgk/dkgb` plus the gate-chain
// outputs `dqg/dkg/dg`.
extern "C" int b2gate_launch(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv, const void* dwin,
                             const void* A4, const void* beta, const void* qin,
                             const void* kin, const void* gin,
                             void* dq, void* dk, void* dw, void* dgk, void* dkgb,
                             void* dqg, void* dkg, void* dg, float scale,
                             int B, int T, int H, int NT, void* stream) {
    return b2mma_launch_impl(v, vn, h, dh, dov, dvv, dwin, dq, dk, dw, dgk,
                             A4, dkgb, beta, qin, kin, gin, dqg, dkg, dg, scale,
                             B, T, H, NT, stream);
}
#endif
#endif  // !B2M_DA && !B2M_DB

#if B2M_DA && !B2M_SAND
// P7-54 dedicated launcher (`b2da_launch`): only valid when compiled with `-DB2M_DA` and
// neither `-DB2M_OUT` nor `-DB2M_DKGB` (it assumes the bare `b2mma_kernel` signature).
static int b2da_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                            const void* dov, const void* dvv, const void* k, const void* g,
                            void* dq, void* dk, void* dw, void* dgk, void* dApre,
                            int B, int T, int H, int NT, void* stream) {
    dim3 grid(NT, B * H);
    int smem = (int)sizeof(Shared);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b2mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
#if B2M_TMA_ON
    CUtensorMap dom, vnm, dvm, hm, dhm, vvm;
    make_map_dov(&dom, (void*)dov, H, T, B);
    make_map_dov(&vnm, (void*)vn,  H, T, B);
    make_map_dov(&dvm, (void*)dvv, H, T, B);
    make_map_dov(&vvm, (void*)v,   H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, vvm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, (float*)dApre,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, (float*)dApre,
        B, T, H, NT);
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2da_launch(const void* v, const void* vn, const void* h, const void* dh,
                           const void* dov, const void* dvv, const void* k, const void* g,
                           void* dq, void* dk, void* dw, void* dgk, void* dApre,
                           int B, int T, int H, int NT, void* stream) {
    return b2da_launch_impl(v, vn, h, dh, dov, dvv, k, g, dq, dk, dw, dgk, dApre,
                            B, T, H, NT, stream);
}
#endif  // B2M_DA && !B2M_SAND

#if B2M_SAND && !B2M_PROD
// P7-55 dedicated launcher (`b2sand_launch`): compiled with `-DB2M_DA -DB2M_SAND`
// (plus the usual stage options).  Outputs both `dApre` (P7-54) and the final `dA`.
static int b2sand_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                              const void* dov, const void* dvv, const void* k, const void* g,
                              const void* A4, const void* beta,
                              void* dq, void* dk, void* dw, void* dgk, void* dApre, void* dA,
                              int B, int T, int H, int NT, void* stream) {
    dim3 grid(NT, B * H);
    int smem = (int)sizeof(Shared);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b2mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
#if B2M_TMA_ON
    CUtensorMap dom, vnm, dvm, hm, dhm, vvm;
    make_map_dov(&dom, (void*)dov, H, T, B);
    make_map_dov(&vnm, (void*)vn,  H, T, B);
    make_map_dov(&dvm, (void*)dvv, H, T, B);
    make_map_dov(&vvm, (void*)v,   H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, vvm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, (float*)dApre,
        (const float*)A4, (const float*)beta, (float*)dA,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, (float*)dApre,
        (const float*)A4, (const float*)beta, (float*)dA,
        B, T, H, NT);
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2sand_launch(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv, const void* k, const void* g,
                             const void* A4, const void* beta,
                             void* dq, void* dk, void* dw, void* dgk, void* dApre, void* dA,
                             int B, int T, int H, int NT, void* stream) {
    return b2sand_launch_impl(v, vn, h, dh, dov, dvv, k, g, A4, beta,
                              dq, dk, dw, dgk, dApre, dA, B, T, H, NT, stream);
}
#endif  // B2M_SAND && !B2M_PROD


#if B2M_DB && !B2M_PROD
// P7-56 dedicated launcher (`b2db_launch`): compiled with `-DB2M_OUT -DB2M_DKGB -DB2M_DB`
// (plus the usual stage options).  Emits `dvb` (P7-52), `dkgb` (P7-53) and the complete `db`.
static int b2db_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                            const void* dov, const void* dvv, const void* dwin,
                            const void* k, const void* g, const void* A4, const void* beta,
                            void* dq, void* dk, void* dw, void* dgk, void* dkgb,
                            void* dv2, void* dvb, void* db_out,
                            int B, int T, int H, int NT, void* stream) {
    dim3 grid(NT, B * H);
    int smem = (int)sizeof(Shared);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b2mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
#if B2M_TMA_ON
    CUtensorMap dom, vnm, dvm, hm, dhm, dwm, vvm;
    make_map_dov(&dom, (void*)dov, H, T, B);
    make_map_dov(&vnm, (void*)vn,  H, T, B);
    make_map_dov(&dvm, (void*)dvv, H, T, B);
    make_map_dov(&dwm, (void*)dwin, H, T, B);
    make_map_dov(&vvm, (void*)v,   H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, dwm, vvm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
        (float*)dkgb, (float*)db_out,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
        (float*)dkgb, (float*)db_out,
        B, T, H, NT);
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2db_launch(const void* v, const void* vn, const void* h, const void* dh,
                           const void* dov, const void* dvv, const void* dwin,
                           const void* k, const void* g, const void* A4, const void* beta,
                           void* dq, void* dk, void* dw, void* dgk, void* dkgb,
                           void* dv2, void* dvb, void* db_out,
                           int B, int T, int H, int NT, void* stream) {
    return b2db_launch_impl(v, vn, h, dh, dov, dvv, dwin, k, g, A4, beta,
                            dq, dk, dw, dgk, dkgb, dv2, dvb, db_out, B, T, H, NT, stream);
}
#endif  // B2M_DB && !B2M_PROD


#if B2M_PROD
// P7-60: the **single production launch** -- all six non-GEMM segments plus the GEMM core in
// one kernel invocation.  Argument order is the exact union of the per-segment kernel tails
// (macro-expanded `b2mma_kernel` signature): maps (dwm, vvm included) / v,vn,h,dh,do,dv /
// k,g / dwin / dq,dk,dw_out,dgk / A4,beta / dv2,dvb / dkgb / q,k,g,dqg,dkg,dg,scale /
// db_out / dApre / A4s,dAout.  The raw `dq/dk` stores are suppressed by the GATE segment
// (production form), so those pointers are scratch.
static int b2prod_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                              const void* dov, const void* dvv,
                              const void* k, const void* g,
                              const void* dwin,
                              void* dq, void* dk, void* dw_out, void* dgk,
                              const void* A4, const void* beta,
                              void* dv2, void* dvb,
                              void* dkgb,
                              const void* qptr, const void* kraw, const void* graw,
                              void* dqg, void* dkg, void* dg, float scale,
                              void* db_out,
                              void* dApre,
                              const void* A4s, void* dAout,
#if B2M_INTRA
                              const void* dAqk_in,
#endif
                              int B, int T, int H, int NT, void* stream) {
    dim3 grid(NT, B * H);
    int smem = (int)sizeof(Shared);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b2mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
#if B2M_TMA_ON
    CUtensorMap dom, vnm, dvm, hm, dhm, dwm, vvm;
    make_map_dov(&dom, (void*)dov,  H, T, B);
    make_map_dov(&vnm, (void*)vn,   H, T, B);
    make_map_dov(&dvm, (void*)dvv,  H, T, B);
    make_map_dov(&dwm, (void*)dwin, H, T, B);
    make_map_dov(&vvm, (void*)v,    H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, dwm, vvm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw_out, (float*)dgk,
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
        (float*)dkgb,
        (const float*)qptr, (const float*)kraw, (const float*)graw,
        (float*)dqg, (float*)dkg, (float*)dg, scale,
        (float*)db_out, (float*)dApre, (const float*)A4s, (float*)dAout,
#if B2M_INTRA
        (const float*)dAqk_in,
#endif
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw_out, (float*)dgk,
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
        (float*)dkgb,
        (const float*)qptr, (const float*)kraw, (const float*)graw,
        (float*)dqg, (float*)dkg, (float*)dg, scale,
        (float*)db_out, (float*)dApre, (const float*)A4s, (float*)dAout,
#if B2M_INTRA
        (const float*)dAqk_in,
#endif
        B, T, H, NT);
#endif
    return (int)cudaGetLastError();
}

#if !B2M_INTRA
extern "C" int b2prod_launch(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv,
                             const void* k, const void* g, const void* dwin,
                             void* dq, void* dk, void* dw_out, void* dgk,
                             const void* A4, const void* beta, void* dv2, void* dvb,
                             void* dkgb, const void* qptr, const void* kraw, const void* graw,
                             void* dqg, void* dkg, void* dg, float scale,
                             void* db_out, void* dApre, const void* A4s, void* dAout,
                             int B, int T, int H, int NT, void* stream) {
    return b2prod_launch_impl(v, vn, h, dh, dov, dvv, k, g, dwin,
                              dq, dk, dw_out, dgk, A4, beta, dv2, dvb,
                              dkgb, qptr, kraw, graw, dqg, dkg, dg, scale,
                              db_out, dApre, A4s, dAout, B, T, H, NT, stream);
}
#else
// P7-63: production launch with the B3 `intra` diagonal folded in (extra `dAqk` input; the
// merged `dq/dk/dg/db` already include the intra contribution).
extern "C" int b2prod_intra_launch(const void* v, const void* vn, const void* h, const void* dh,
                                   const void* dov, const void* dvv,
                                   const void* k, const void* g, const void* dwin,
                                   void* dq, void* dk, void* dw_out, void* dgk,
                                   const void* A4, const void* beta, void* dv2, void* dvb,
                                   void* dkgb, const void* qptr, const void* kraw,
                                   const void* graw,
                                   void* dqg, void* dkg, void* dg, float scale,
                                   void* db_out, void* dApre, const void* A4s, void* dAout,
                                   const void* dAqk, int B, int T, int H, int NT,
                                   void* stream) {
    return b2prod_launch_impl(v, vn, h, dh, dov, dvv, k, g, dwin,
                              dq, dk, dw_out, dgk, A4, beta, dv2, dvb,
                              dkgb, qptr, kraw, graw, dqg, dkg, dg, scale,
                              db_out, dApre, A4s, dAout, dAqk, B, T, H, NT, stream);
}
#endif
#endif  // B2M_PROD


extern "C" int b2mma_smem_bytes() { return (int)sizeof(Shared); }
