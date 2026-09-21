// 43 W4A16 跨 item 持久化流水（主题 26c）
//
// 承接 41 篇：41 用 warp specialization 把反量化与 wgmma 拆到不同 warpgroup，
// occupancy 6.5%→21%，但只快 1.07×；ncu 指出真正的墙是 wave quantization：
// grid=680 CTA / 264 槽 = 3 波（末波 152 个 CTA），Est. Speedup 33%。
// 41 试过 persistent + atomicAdd 动态派活，但每换一个 work item 就重启
// cp.async/wgmma 流水（重新 prologue、drain、barrier 相位），收益被吃光。
//
// 本篇：**把 (item, kb) 展平成一维 stage 流**，一个 CTA 在多个 work item 之间
// **不停机**地跑同一条 producer/consumer 流水：
//   - producer 顺序 load+dequant 展平后的每个 stage，永不 drain；
//   - consumer 顺序 wgmma，遇到 item 边界才做一次 epilogue（挪出热循环）；
//   - 累加器清零改用 wgmma 自身的 scale_d=0（37 篇 W1），避免 ptxas C7514。
// work item = (n_tile, k-chunk)，CHUNK 个 BK（偶数，且整除 nblk），静态
// round-robin 分给 P=2*NSM 个 persistent CTA。stage→(item,kb) 用闭式映射
// （t/CHUNK、t%CHUNK），无需任何 per-stage 元数据数组。
//
// 真实 shape 取自 /ssd/models/qwen3-8B/config.json：
//   hidden_size=5120, intermediate_size=17408, group_size=128, 对称 int4。
// 默认测 M=64（decode）× N=17408 × K=5120。
//
// 运行：
//   ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//     scripts/run.sh 43-w4a16-persist/w4a16_persist.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int GROUP = 128;
constexpr int BK = 64;  // bf16 SW128 atom：64 元素 = 128B
constexpr double BF16_PEAK = 989.0;
constexpr int NSM = 132;

// ---------------------------------------------------------------------------
// PTX helpers（与 41 篇同）
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

// scale_d：k>0 累加；k=0 走 ScaleOut::Zero（用指令自己清零累加器，见 37 篇 W1）
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t da, uint64_t db, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(scale_d));
}

// bf16 SW128 K-major：[row/8][k/64][8][64]，atom 1024B，16B 列 c'=c^r
__device__ __forceinline__ int sw128_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int c = kk >> 3, cc = c ^ rr;
  return (rg * (K >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;  // LBO = 1 (16B)
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}
__device__ __forceinline__ uint32_t k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}

// mbarrier + named barrier
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred p;\nLAB_WAIT%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(phase));
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// ---------------------------------------------------------------------------
// (A) 41 篇的非持久化 baseline（同进程 head-to-head）
// ---------------------------------------------------------------------------
template <int BM, int BN, int STAGES, int KSPLIT, int NCONS, bool DQ = true, int NPRODW = 1,
          int MINB = 1>
__global__ void __launch_bounds__((NPRODW + NCONS) * 128, MINB) w4a16_ws_kernel(
    const bf16* __restrict__ A, const uint8_t* __restrict__ Wp, const float* __restrict__ Ws,
    float* __restrict__ C, int M, int N, int K) {
  static_assert(BM == 64, "BM 固定 64");
  constexpr int NPB = BK / 2;
  constexpr int NPROD = NPRODW * 128;
  constexpr int T = (NPRODW + NCONS) * 128;
  constexpr int SBO = 1024;
  constexpr int ABYTES = BM * BK * 2;
  constexpr int WB_BYTES = BN * NPB;
  constexpr int SW_BYTES = BN * BK * 2;

  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* As = smem + ((2 * STAGES * 8 + 1023) / 1024) * 1024;
  char* Wps = As + (size_t)STAGES * ABYTES;
  char* sW = Wps + (size_t)STAGES * WB_BYTES;
  float* sctab = reinterpret_cast<float*>(sW + (size_t)STAGES * SW_BYTES);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int nblk = K / BK;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;
  const int nblk_z = kb1 - kb0;
  const int ngrp = K / GROUP;
  const int NGR = (nblk_z + 1) >> 1;

  for (int s = tid; s < STAGES; s += T) {
    mbar_init(full + s, 1);
    mbar_init(empty + s, NCONS * 128);
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();

  if (tid < NPROD) {
    const int g0 = kb0 >> 1;
    const int ngrp_rel = (nblk_z + 1) >> 1;
    for (int i = tid; i < BN * ngrp_rel; i += NPROD) {
      const int n = i / ngrp_rel, gg = i % ngrp_rel;
      sctab[n * NGR + gg] = __ldg(&Ws[(size_t)(block_col + n) * ngrp + g0 + gg]);
    }
    auto load_stage = [&](int st, int kb) {
      const int k0 = kb * BK;
      char* a = As + (size_t)st * ABYTES;
      char* wp = Wps + (size_t)st * WB_BYTES;
      for (int i = tid; i < BM * (BK / 8); i += NPROD) {
        const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
        __pipeline_memcpy_async(a + sw128_off(r, c8, BK), &A[(size_t)(block_row + r) * K + k0 + c8], 16);
      }
      for (int i = tid; i < BN * (NPB / 16); i += NPROD) {
        const int r = i / (NPB / 16), c = (i % (NPB / 16)) * 16;
        __pipeline_memcpy_async(wp + r * NPB + c, &Wp[(size_t)(block_col + r) * (K / 2) + k0 / 2 + c], 16);
      }
      __pipeline_commit();
    };
    auto dequant_stage = [&](int st, int kb) {
      if constexpr (!DQ) return;
      char* w = sW + (size_t)st * SW_BYTES;
      const int gg = (kb - kb0) >> 1;
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(Wps + (size_t)st * WB_BYTES);
      constexpr int M4 = NPB / 4;
      for (int i = tid; i < BN * M4; i += NPROD) {
        const int n = i / M4, m4 = i % M4;
        const uint32_t word = wp[i];
        const float s = sctab[n * NGR + gg];
        bf16 out[8];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int q0 = (int)((word >> (8 * b)) & 0xF) - 8;
          const int q1 = (int)((word >> (8 * b + 4)) & 0xF) - 8;
          out[2 * b] = __float2bfloat16((float)q0 * s);
          out[2 * b + 1] = __float2bfloat16((float)q1 * s);
        }
        const int kk = (8 * m4) & 63;
        const int rr = n & 7, rg = n >> 3;
        const int cc = (kk >> 3) ^ rr;
        const int off = rg * 1024 + (rr * 8 + cc) * 16;
        *reinterpret_cast<uint4*>(w + off) = *reinterpret_cast<const uint4*>(out);
      }
    };
    constexpr int PF = 1;
#pragma unroll
    for (int p = 0; p < PF; ++p) {
      if (p < nblk_z) load_stage(p, kb0 + p);
      else __pipeline_commit();
    }
    for (int t = 0; t < nblk_z; ++t) {
      const int s = t % STAGES;
      const int tn = t + PF;
      if (tn < nblk_z) {
        if (tn >= STAGES) {
          const uint32_t ph = (uint32_t)((tn / STAGES - 1) & 1);
          mbar_wait(empty + tn % STAGES, ph);
        }
        load_stage(tn % STAGES, kb0 + tn);
      } else {
        __pipeline_commit();
      }
      __pipeline_wait_prior(PF);
      named_bar_sync(1, NPROD);
      dequant_stage(s, kb0 + t);
      named_bar_sync(1, NPROD);
      fence_proxy_async();
      if (tid == 0) mbar_arrive(full + s);
    }
  } else {
    const int cg = (warp - NPRODW * 4) >> 2;
    const int nbase = cg * 128;
    float acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;
    char* myw = sW + (size_t)nbase * BK * 2;

    for (int t = 0; t < nblk_z; ++t) {
      const int s = t % STAGES;
      mbar_wait(full + s, (uint32_t)((t / STAGES) & 1));
      char* a = As + (size_t)s * ABYTES;
      char* w = myw + (size_t)s * SW_BYTES;
      wgmma_fence();
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        uint64_t da = make_desc_sw128(k16_addr(smem_u32(a), kk), SBO);
        uint64_t db = make_desc_sw128(k16_addr(smem_u32(w), kk), SBO);
        wgmma_m64n128k16(acc, da, db, 1);
      }
      wgmma_commit();
      wgmma_wait0();
      mbar_arrive(empty + s);
    }
    wgmma_wait0();

    const int row0 = (warp & 3) * 16 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = block_col + nbase + j * 8 + (lane & 3) * 2;
      const int r0 = block_row + row0, r1 = r0 + 8;
      if (KSPLIT == 1) {
        if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + col]) = make_float2(acc[j * 4 + 0], acc[j * 4 + 1]);
        if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + col]) = make_float2(acc[j * 4 + 2], acc[j * 4 + 3]);
      } else {
        if (r0 < M) {
          atomicAdd(&C[(size_t)r0 * N + col], acc[j * 4 + 0]);
          atomicAdd(&C[(size_t)r0 * N + col + 1], acc[j * 4 + 1]);
        }
        if (r1 < M) {
          atomicAdd(&C[(size_t)r1 * N + col], acc[j * 4 + 2]);
          atomicAdd(&C[(size_t)r1 * N + col + 1], acc[j * 4 + 3]);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// (B) 43 篇：持久化 + 跨 item 连续流水
//   BM=64；BN=NCONS*128；threads=(1+NCONS)*128。
//   work item = (n_tile, k-chunk)，每个 item 恰好 CHUNK 个 BK；
//   要求 nblk % CHUNK == 0（nblk=K/BK）。stage→item 用闭式映射。
// ---------------------------------------------------------------------------
template <int BM, int BN, int STAGES, int CHUNK, int NCONS, bool DQ = true, int NPRODW = 1,
          int MINB = 2>
__global__ void __launch_bounds__((NPRODW + NCONS) * 128, MINB) w4a16_persist_kernel(
    const bf16* __restrict__ A, const uint8_t* __restrict__ Wp, const float* __restrict__ Ws,
    float* __restrict__ C, int M, int N, int K, int total_items, int P) {
  static_assert(BM == 64, "BM 固定 64");
  static_assert(CHUNK % 2 == 0, "CHUNK 必须偶数（group=128=2*BK）");
  static_assert(CHUNK >= STAGES, "CHUNK < STAGES 会破坏跨 item 流水（实测 illegal memory access）");
  constexpr int NPB = BK / 2;
  constexpr int NPROD = NPRODW * 128;
  constexpr int T = (NPRODW + NCONS) * 128;
  constexpr int SBO = 1024;
  constexpr int ABYTES = BM * BK * 2;
  constexpr int WB_BYTES = BN * NPB;
  constexpr int SW_BYTES = BN * BK * 2;
  constexpr int NGR = CHUNK / 2;  // 每个 item 的 group 数（CHUNK 偶数）

  extern __shared__ __align__(1024) char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  char* base = smem + ((2 * STAGES * 8 + 1023) / 1024) * 1024;
  char* As = base;
  char* Wps = As + (size_t)STAGES * ABYTES;
  char* sW = Wps + (size_t)STAGES * WB_BYTES;
  float* sctab = reinterpret_cast<float*>(sW + (size_t)STAGES * SW_BYTES);  // [BN][NGR]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ntiles = N / BN;
  const int nblk = K / BK;
  const int ngrp = K / GROUP;
  const int gid0 = blockIdx.x;
  // 本 CTA 的 item 数（gid = gid0, gid0+P, ... < total_items）
  const int nitems = (gid0 < total_items) ? (total_items - 1 - gid0) / P + 1 : 0;
  const int NS = nitems * CHUNK;

  for (int s = tid; s < STAGES; s += T) {
    mbar_init(full + s, 1);
    mbar_init(empty + s, NCONS * 128);
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();

  if (tid < NPROD) {
    // ================= PRODUCER =================
    constexpr int PF = 1;
    auto load_flat = [&](int t) {
      const int ii = t / CHUNK, u = t % CHUNK;
      const int gid = gid0 + ii * P;
      const int col = (gid % ntiles) * BN;
      const int k0 = ((gid / ntiles) * CHUNK + u) * BK;
      const int s = t % STAGES;
      char* a = As + (size_t)s * ABYTES;
      char* wp = Wps + (size_t)s * WB_BYTES;
      for (int i = tid; i < BM * (BK / 8); i += NPROD) {
        const int r = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
        __pipeline_memcpy_async(a + sw128_off(r, c8, BK), &A[(size_t)r * K + k0 + c8], 16);
      }
      for (int i = tid; i < BN * (NPB / 16); i += NPROD) {
        const int r = i / (NPB / 16), c = (i % (NPB / 16)) * 16;
        __pipeline_memcpy_async(wp + r * NPB + c, &Wp[(size_t)(col + r) * (K / 2) + k0 / 2 + c], 16);
      }
      __pipeline_commit();
    };
    auto dequant_flat = [&](int t) {
      if constexpr (!DQ) return;
      const int u = t % CHUNK;
      const int s = t % STAGES;
      char* w = sW + (size_t)s * SW_BYTES;
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(Wps + (size_t)s * WB_BYTES);
      const int gg = u >> 1;  // 相对 group 下标（kb=kb0+u，kb0 偶数）
      constexpr int M4 = NPB / 4;
      for (int i = tid; i < BN * M4; i += NPROD) {
        const int n = i / M4, m4 = i % M4;
        const uint32_t word = wp[i];
        const float scl = sctab[n * NGR + gg];
        bf16 out[8];
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          const int q0 = (int)((word >> (8 * b)) & 0xF) - 8;
          const int q1 = (int)((word >> (8 * b + 4)) & 0xF) - 8;
          out[2 * b] = __float2bfloat16((float)q0 * scl);
          out[2 * b + 1] = __float2bfloat16((float)q1 * scl);
        }
        const int kk = (8 * m4) & 63;
        const int rr = n & 7, rg = n >> 3;
        const int cc = (kk >> 3) ^ rr;
        const int off = rg * 1024 + (rr * 8 + cc) * 16;
        *reinterpret_cast<uint4*>(w + off) = *reinterpret_cast<const uint4*>(out);
      }
    };

#pragma unroll
    for (int pr = 0; pr < PF; ++pr) {
      if (pr < NS) load_flat(pr);
      else __pipeline_commit();
    }
    for (int t = 0; t < NS; ++t) {
      const int s = t % STAGES;
      const int tn = t + PF;
      if (tn < NS) {
        if (tn >= STAGES) mbar_wait(empty + tn % STAGES, (uint32_t)((tn / STAGES - 1) & 1));
        load_flat(tn);
      } else {
        __pipeline_commit();
      }
      if ((t % CHUNK) == 0) {  // item 起点：预取该 item 的 per-group scale
        const int gid = gid0 + (t / CHUNK) * P;
        const int col = (gid % ntiles) * BN;
        const int kb0 = (gid / ntiles) * CHUNK;
        for (int i = tid; i < BN * NGR; i += NPROD) {
          const int n = i / NGR, gg = i % NGR;
          sctab[n * NGR + gg] = __ldg(&Ws[(size_t)(col + n) * ngrp + (kb0 >> 1) + gg]);
        }
      }
      __pipeline_wait_prior(PF);
      named_bar_sync(1, NPROD);
      dequant_flat(t);
      named_bar_sync(1, NPROD);
      fence_proxy_async();
      if (tid == 0) mbar_arrive(full + s);
    }
  } else {
    // ================= CONSUMER =================
    const int cg = (warp - NPRODW * 4) >> 2;
    const int nbase = cg * 128;
    const int nbase_k = nbase * BK * 2;
    const int row0 = (warp & 3) * 16 + (lane >> 2);
    float acc[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) acc[i] = 0.f;

    int t = 0;
    for (int ii = 0; ii < nitems; ++ii) {
      const int gid = gid0 + ii * P;
      const int col = (gid % ntiles) * BN + nbase;
#pragma unroll 1
      for (int u = 0; u < CHUNK; ++u, ++t) {
        const int s = t % STAGES;
        mbar_wait(full + s, (uint32_t)((t / STAGES) & 1));
        char* a = As + (size_t)s * ABYTES;
        char* w = sW + (size_t)(s * SW_BYTES + nbase_k);
        wgmma_fence();
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
          uint64_t da = make_desc_sw128(k16_addr(smem_u32(a), kk), SBO);
          uint64_t db = make_desc_sw128(k16_addr(smem_u32(w), kk), SBO);
          wgmma_m64n128k16(acc, da, db, (u == 0 && kk == 0) ? 0 : 1);
        }
        wgmma_commit();
        wgmma_wait0();
        mbar_arrive(empty + s);
      }
      // item 边界 epilogue（热循环之外；下一 item 由 scale_d=0 清零累加器）
#pragma unroll
      for (int q = 0; q < 16; ++q) {
        const int c = col + q * 8 + (lane & 3) * 2;
        const int r0 = row0, r1 = r0 + 8;
        if (CHUNK >= nblk) {
          if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + c]) = make_float2(acc[q * 4 + 0], acc[q * 4 + 1]);
          if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + c]) = make_float2(acc[q * 4 + 2], acc[q * 4 + 3]);
        } else {
          if (r0 < M) {
            atomicAdd(&C[(size_t)r0 * N + c], acc[q * 4 + 0]);
            atomicAdd(&C[(size_t)r0 * N + c + 1], acc[q * 4 + 1]);
          }
          if (r1 < M) {
            atomicAdd(&C[(size_t)r1 * N + c], acc[q * 4 + 2]);
            atomicAdd(&C[(size_t)r1 * N + c + 1], acc[q * 4 + 3]);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
static bf16 f2b(float x) { return __float2bfloat16(x); }

static size_t persist_smem(int BN, int STAGES, int CHUNK) {
  const int NGR = CHUNK / 2;
  size_t scal = 2 * STAGES * 8 + (size_t)BN * NGR * 4;
  size_t base = (scal + 1023) / 1024 * 1024;
  size_t tens = (size_t)STAGES * (64 * BK * 2 + (size_t)BN * (BK / 2) + (size_t)BN * BK * 2);
  return base + tens;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int M = (argc > 1) ? std::atoi(argv[1]) : 64;
  int N = (argc > 2) ? std::atoi(argv[2]) : 17408;
  int K = (argc > 3) ? std::atoi(argv[3]) : 5120;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  const double w_bytes_int4 = (double)N * K / 2;
  std::printf("\nW4A16 persistent GEMM: M=%d N=%d K=%d group=%d\n", M, N, K, GROUP);
  std::printf("FLOPs = %.2f GFLOP ; W_int4 = %.1f MB\n\n", flops / 1e9, w_bytes_int4 / 1e6);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<bf16> hA((size_t)M * K);
  std::vector<float> hWf((size_t)N * K), hWdeq((size_t)N * K);
  std::vector<uint8_t> hWp((size_t)N * (K / 2));
  std::vector<float> hWs((size_t)N * (K / GROUP));
  std::vector<float> hC((size_t)M * N);
  for (auto& v : hA) v = f2b(0.5f * dist(rng));
  for (auto& v : hWf) v = 0.1f * dist(rng);
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < K / GROUP; ++g) {
      float amax = 0.f;
      for (int k = 0; k < GROUP; ++k) amax = std::max(amax, std::fabs(hWf[(size_t)n * K + g * GROUP + k]));
      const float s = amax > 0 ? amax / 7.f : 1.f;
      hWs[(size_t)n * (K / GROUP) + g] = s;
      for (int k = 0; k < GROUP; ++k) {
        const size_t idx = (size_t)n * K + g * GROUP + k;
        int q = std::max(-8, std::min(7, (int)std::lround(hWf[idx] / s)));
        hWdeq[idx] = q * s;
        const size_t pk = (size_t)n * (K / 2) + (g * GROUP + k) / 2;
        if (((g * GROUP + k) & 1) == 0) hWp[pk] = (uint8_t)((hWp[pk] & 0xF0) | ((q + 8) & 0xF));
        else hWp[pk] = (uint8_t)((hWp[pk] & 0x0F) | (((q + 8) & 0xF) << 4));
      }
    }

  bf16* dA; uint8_t* dWp; float *dWs, *dC;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
  CUDA_CHECK(cudaMalloc(&dWp, hWp.size()));
  CUDA_CHECK(cudaMalloc(&dWs, hWs.size() * 4));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWp, hWp.data(), hWp.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWs, hWs.data(), hWs.size() * 4, cudaMemcpyHostToDevice));

  auto cpu_ref = [&](int m, int n) {
    double s = 0;
    for (int k = 0; k < K; ++k) s += (double)__bfloat162float(hA[(size_t)m * K + k]) * (double)hWdeq[(size_t)n * K + k];
    return s;
  };
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 32; ++i) {
      const int m = (i * 1009 + 13) % M, n = (i * 997 + 7) % N;
      const double e = cpu_ref(m, n);
      err = std::max(err, std::fabs((double)hC[(size_t)m * N + n] - e));
      ref = std::max(ref, std::fabs(e));
    }
    const bool ok = err / std::max(ref, 1e-6) < 3e-2;
    std::printf("  [%-22s] max_abs_err=%.3e (ref~%.2f) %s\n", tag, err, ref, ok ? "OK" : "FAIL");
    return ok;
  };
  auto report = [&](const char* tag, double ms) {
    const double tf = to_tflops(flops, ms);
    const double wbw = w_bytes_int4 * 1000.0 / ms / 1e9;
    std::printf("%-24s %9.4f ms  %8.2f TFLOPS (%5.1f%% bf16)  W-bw %7.1f GB/s\n", tag, ms, tf, 100.0 * tf / BF16_PEAK, wbw);
  };
  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };

#define RUN_BASE(NAME, ST, KS, MB)                                                                  \
    do {                                                                                            \
      auto fn = w4a16_ws_kernel<64, 128, ST, KS, 1, true, 1, MB>;                                   \
      const int nblk_ = K / BK, ngrmax = (nblk_ / KS + 1) / 2 + 1;                                  \
      const size_t shm = 1024 + (size_t)ST * (64 * BK * 2 + 128 * (BK / 2) + 128 * BK * 2) +        \
                         128 * ngrmax * 4;                                                          \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));  \
      dim3 grid(div_up(N, 128), div_up(M, 64), KS);                                                 \
      auto launch = [&] {                                                                           \
        cudaMemsetAsync(dC, 0, (size_t)M * N * 4);                                                  \
        fn<<<grid, 256, shm>>>(dA, dWp, dWs, dC, M, N, K);                                          \
      };                                                                                            \
      launch();                                                                                     \
      CUDA_CHECK_LAST();                                                                            \
      check(NAME);                                                                                  \
      report(NAME, bench_ms(launch, 5, 50));                                                        \
    } while (0)

  if (want("base")) {
    RUN_BASE("base_41_k5_s3", 3, 5, 1);  // 41 篇原配（2 CTA/SM）
    RUN_BASE("base_s2_k5_mb2", 2, 5, 2); // STAGES=2，仍 2 CTA/SM
    RUN_BASE("base_s2_k4_mb2", 2, 4, 2);
    RUN_BASE("base_s2_k8_mb2", 2, 8, 2);
#ifdef TRY3CTA
    RUN_BASE("base_s2_k5_mb3", 2, 5, 3);  // 复现 3 CTA/SM：ptxas C7602 Insufficient registers
#endif
  }

  if (want("cmp")) {  // M 扫描用：baseline KSPLIT=5 vs persistent CHUNK=4/8/16
    RUN_BASE("cmp_base_k5", 3, 5, 1);
    RUN_BASE("cmp_base_k8", 3, 8, 1);
    RUN_BASE("cmp_base_k16", 3, 16, 1);
#define RUN_P1(NAME, ST, CH, NC)                                                                    \
    do {                                                                                            \
      auto fn = w4a16_persist_kernel<64, 128, ST, CH, NC>;                                          \
      const int total_items = (N / 128) * ((K / BK) / (CH));                                         \
      const int P = 2 * NSM;                                                                        \
      const size_t shm = persist_smem(128, ST, CH);                                                 \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));  \
      auto launch = [&] {                                                                           \
        cudaMemsetAsync(dC, 0, (size_t)M * N * 4);                                                  \
        fn<<<P, 256, shm>>>(dA, dWp, dWs, dC, M, N, K, total_items, P);                            \
      };                                                                                            \
      launch(); CUDA_CHECK_LAST(); check(NAME);                                                     \
      report(NAME, bench_ms(launch, 5, 50));                                                        \
    } while (0)
    RUN_P1("cmp_persist_c4", 3, 4, 1);
    RUN_P1("cmp_persist_c8", 3, 8, 1);
    RUN_P1("cmp_persist_c16", 3, 16, 1);
  }

  if (want("persist")) {
#define RUN_P(NAME, ST, CH, NC)                                                                     \
    do {                                                                                            \
      if ((K / BK) % (CH) != 0) { std::printf("%-24s SKIP (nblk%%CHUNK!=0)\n", NAME); break; }       \
      auto fn = w4a16_persist_kernel<64, 128, ST, CH, NC>;                                          \
      const int total_items = (N / 128) * ((K / BK) / (CH));                                         \
      const int P = 2 * NSM;                                                                        \
      const size_t shm = persist_smem(128, ST, CH);                                                 \
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));  \
      const int nthr = (1 + NC) * 128;                                                              \
      auto launch = [&] {                                                                           \
        cudaMemsetAsync(dC, 0, (size_t)M * N * 4);                                                  \
        fn<<<P, nthr, shm>>>(dA, dWp, dWs, dC, M, N, K, total_items, P);                            \
      };                                                                                            \
      launch();                                                                                     \
      CUDA_CHECK_LAST();                                                                            \
      check(NAME);                                                                                  \
      report(NAME, bench_ms(launch, 5, 50));                                                        \
    } while (0)

    RUN_P("persist_s3c16", 3, 16, 1);
    RUN_P("persist_s3c8", 3, 8, 1);
    RUN_P("persist_s3c4", 3, 4, 1);
    RUN_P("persist_s4c16", 4, 16, 1);
    RUN_P("persist_s4c8", 4, 8, 1);
    RUN_P("persist_s2c16", 2, 16, 1);
    RUN_P("persist_s5c16", 5, 16, 1);
  }

  if (want("probe_base")) {
    auto fn = w4a16_ws_kernel<64, 128, 3, 5, 1>;
    const int nblk_ = K / BK, ngrmax = (nblk_ / 5 + 1) / 2 + 1;
    const size_t shm = 1024 + 3 * (64 * BK * 2 + 128 * (BK / 2) + 128 * BK * 2) + 128 * ngrmax * 4;
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    dim3 grid(div_up(N, 128), div_up(M, 64), 5);
    auto launch = [&] {
      cudaMemsetAsync(dC, 0, (size_t)M * N * 4);
      fn<<<grid, 256, shm>>>(dA, dWp, dWs, dC, M, N, K);
    };
    launch(); CUDA_CHECK_LAST(); check("probe_base"); report("probe_base", bench_ms(launch, 5, 50));
  }

  if (want("probe_c8")) {
    auto fn = w4a16_persist_kernel<64, 128, 3, 8, 1>;
    const int total_items = (N / 128) * ((K / BK) / 8);
    const int P = 2 * NSM;
    const size_t shm = persist_smem(128, 3, 8);
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    auto launch = [&] {
      cudaMemsetAsync(dC, 0, (size_t)M * N * 4);
      fn<<<P, 256, shm>>>(dA, dWp, dWs, dC, M, N, K, total_items, P);
    };
    launch(); CUDA_CHECK_LAST(); check("probe_c8"); report("probe_c8", bench_ms(launch, 5, 50));
  }

  if (want("probe")) {
    auto fn = w4a16_persist_kernel<64, 128, 3, 16, 1>;
    const int total_items = (N / 128) * ((K / BK) / 16);
    const int P = 2 * NSM;
    const size_t shm = persist_smem(128, 3, 16);
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
    auto launch = [&] {
      cudaMemsetAsync(dC, 0, (size_t)M * N * 4);
      fn<<<P, 256, shm>>>(dA, dWp, dWs, dC, M, N, K, total_items, P);
    };
    launch();
    CUDA_CHECK_LAST();
    check("probe_persist");
    report("probe_persist", bench_ms(launch, 5, 50));
  }

  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dWp)); CUDA_CHECK(cudaFree(dWs)); CUDA_CHECK(cudaFree(dC));
  return 0;
}
