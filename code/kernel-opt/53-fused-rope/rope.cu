// 53 融合 RoPE（主题 18）：rotary position embedding + YaRN 频率缩放
//
// 真实语义取自 `/ssd/models/DeepSeek-V4-Pro/inference/model.py`：
//   precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow)
//     -> freqs_cis = polar(1, outer(t, freqs))，freqs 带 YaRN 线性 ramp
//   apply_rotary_emb(q[..., -rd:], freqs_cis)（rd=rope_head_dim=64）
//     -> 用 `view_as_complex` = **交错配对**（x[2j], x[2j+1]），
//        即 angle = pos * (1/base^(2j/dim))，out = x*cos + rot90(x)*sin
//
// 对照 Qwen3/HF：`rotate_half` 是**对半配对**（x[j], x[j+d/2]），theta=1e6，无 YaRN。
// 两种配对方式决定了内存访问的向量化方式，是本文的核心。
//
// 编译/运行：
//   scripts/run.sh 53-fused-rope/rope.cu bench
//   scripts/run.sh 53-fused-rope/rope.cu bench <scenario>
//   scripts/run.sh 53-fused-rope/rope.cu dump <scenario>   # 导出 Xin/Xout 供 python 对拍
//
// 视角：把「读 rope 区 → 旋转 → 写回」做到贴 HBM。绳区只占 head_dim 的一小段
// （DeepSeek 64/512），所以有效带宽 = 2·T·H·rd·2B 的搬运效率。

#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------------------
// 频率表（host，fp64 计算后存 float）—— 完全复刻 model.py 的 precompute_freqs_cis
// ---------------------------------------------------------------------------
static std::vector<float> build_freqs(int R, float base, int original_seq_len, float factor,
                                      int beta_fast, int beta_slow) {
  const int nd = R / 2;
  std::vector<double> f(nd);
  for (int i = 0; i < nd; ++i) f[i] = std::pow((double)base, -2.0 * i / R);

  if (original_seq_len > 0) {
    auto find_dim = [&](double nr) {
      return R * std::log((double)original_seq_len / (nr * 2.0 * M_PI)) /
             (2.0 * std::log((double)base));
    };
    int low = (int)std::floor(find_dim((double)beta_fast));
    int high = (int)std::ceil(find_dim((double)beta_slow));
    low = std::max(low, 0);
    high = std::min(high, R - 1);
    if (low >= high) high = low + 1;
    for (int i = 0; i < nd; ++i) {
      double lin = ((double)i - low) / (high - low);
      double ramp = std::min(std::max(lin, 0.0), 1.0);
      double smooth = 1.0 - ramp;
      f[i] = f[i] / factor * (1.0 - smooth) + f[i] * smooth;
    }
  }
  std::vector<float> out(nd);
  for (int i = 0; i < nd; ++i) out[i] = (float)f[i];
  return out;
}

static std::vector<float2> build_cs(const std::vector<float>& freq, int maxpos) {
  const int nd = (int)freq.size();
  std::vector<float2> cs((size_t)maxpos * nd);
  for (int p = 0; p < maxpos; ++p)
    for (int j = 0; j < nd; ++j) {
      double a = (double)p * freq[j];
      cs[(size_t)p * nd + j] = make_float2((float)cos(a), (float)sin(a));
    }
  return cs;
}

// ---------------------------------------------------------------------------
// 设备端：读 (cos,sin)
// ---------------------------------------------------------------------------
template <bool BF16TAB>
__device__ __forceinline__ float2 read_cs(const void* cs, int idx) {
  if constexpr (BF16TAB) {
    __nv_bfloat162 v = reinterpret_cast<const __nv_bfloat162*>(cs)[idx];
    return make_float2(__bfloat162float(v.x), __bfloat162float(v.y));
  } else {
    return reinterpret_cast<const float2*>(cs)[idx];
  }
}

__device__ __forceinline__ float2 rot(float x0, float x1, float2 cs) {
  return make_float2(x0 * cs.x - x1 * cs.y, x0 * cs.y + x1 * cs.x);
}

// ---------------------------------------------------------------------------
// v0：朴素「一线程一 pair」+ 现场算 powf/sincos + YaRN（最贵）
// ---------------------------------------------------------------------------
template <int R, int PAIR>
__global__ void rope_naive_kernel(bf16* __restrict__ X, int T, int H, int HD,
                                  const int* __restrict__ pos, float base, float factor,
                                  int original_seq_len, int beta_fast, int beta_slow) {
  const int NP = R / 2;
  const long total = (long)T * H * NP;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const int j = (int)(i % NP);
    const long row = i / NP;  // t*H + h
    const long t = row / H;
    const int p = pos ? pos[t] : (int)t;
    double theta = std::pow((double)base, -2.0 * j / R);
    if (original_seq_len > 0) {
      auto find_dim = [&](double nr) {
        return R * std::log((double)original_seq_len / (nr * 2.0 * M_PI)) /
               (2.0 * std::log((double)base));
      };
      int low = (int)floor(find_dim((double)beta_fast));
      int high = (int)ceil(find_dim((double)beta_slow));
      low = low < 0 ? 0 : low;
      high = high > R - 1 ? R - 1 : high;
      if (low >= high) high = low + 1;
      double lin = ((double)j - low) / (high - low);
      double ramp = lin < 0.0 ? 0.0 : (lin > 1.0 ? 1.0 : lin);
      double smooth = 1.0 - ramp;
      theta = theta / factor * (1.0 - smooth) + theta * smooth;
    }
    double a = (double)p * theta;
    float c = (float)cos(a), s = (float)sin(a);
    bf16* rowp = X + row * (long)HD + (HD - R);
    int i0, i1;
    if constexpr (PAIR == 0) { i0 = 2 * j; i1 = 2 * j + 1; } else { i0 = j; i1 = j + NP; }
    float x0 = __bfloat162float(rowp[i0]), x1 = __bfloat162float(rowp[i1]);
    float2 y = rot(x0, x1, make_float2(c, s));
    rowp[i0] = __float2bfloat16(y.x);
    rowp[i1] = __float2bfloat16(y.y);
  }
}

// ---------------------------------------------------------------------------
// v1：一线程一 pair，查表；交错用一次 4B 读写
// ---------------------------------------------------------------------------
template <int R, int PAIR, bool BF16TAB>
__global__ void rope_tab_kernel(bf16* __restrict__ X, int T, int H, int HD,
                                const int* __restrict__ pos, const void* __restrict__ cs) {
  const int NP = R / 2;
  const long total = (long)T * H * NP;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const int j = (int)(i % NP);
    const long row = i / NP;
    const long t = row / H;
    const int p = pos ? pos[t] : (int)t;
    float2 c = read_cs<BF16TAB>(cs, p * NP + j);
    bf16* rowp = X + row * (long)HD + (HD - R);
    if constexpr (PAIR == 0) {
      uint32_t w = *reinterpret_cast<uint32_t*>(rowp + 2 * j);
      float x0 = __bfloat162float(reinterpret_cast<bf16*>(&w)[0]);
      float x1 = __bfloat162float(reinterpret_cast<bf16*>(&w)[1]);
      float2 y = rot(x0, x1, c);
      uint32_t o;
      reinterpret_cast<bf16*>(&o)[0] = __float2bfloat16(y.x);
      reinterpret_cast<bf16*>(&o)[1] = __float2bfloat16(y.y);
      *reinterpret_cast<uint32_t*>(rowp + 2 * j) = o;
    } else {
      float x0 = __bfloat162float(rowp[j]), x1 = __bfloat162float(rowp[j + NP]);
      float2 y = rot(x0, x1, c);
      rowp[j] = __float2bfloat16(y.x);
      rowp[j + NP] = __float2bfloat16(y.y);
    }
  }
}

// ---------------------------------------------------------------------------
// v2/v3：一 warp 一行 (token,head)，lane 沿 pair 维；表 fp32 / bf16
// ---------------------------------------------------------------------------
template <int R, int PAIR, bool BF16TAB>
__global__ void __launch_bounds__(256) rope_warp_kernel(
    bf16* __restrict__ X, int T, int H, int HD, const int* __restrict__ pos,
    const void* __restrict__ cs) {
  const int NP = R / 2;
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const long total = (long)T * H;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (long idx = warp; idx < total; idx += nwarp) {
    const long t = idx / H;
    const int p = pos ? pos[t] : (int)t;
    bf16* rowp = X + idx * (long)HD + (HD - R);
#pragma unroll
    for (int j = lane; j < NP; j += 32) {
      float2 c = read_cs<BF16TAB>(cs, p * NP + j);
      if constexpr (PAIR == 0) {
        uint32_t w = *reinterpret_cast<uint32_t*>(rowp + 2 * j);
        float x0 = __bfloat162float(reinterpret_cast<bf16*>(&w)[0]);
        float x1 = __bfloat162float(reinterpret_cast<bf16*>(&w)[1]);
        float2 y = rot(x0, x1, c);
        uint32_t o;
        reinterpret_cast<bf16*>(&o)[0] = __float2bfloat16(y.x);
        reinterpret_cast<bf16*>(&o)[1] = __float2bfloat16(y.y);
        *reinterpret_cast<uint32_t*>(rowp + 2 * j) = o;
      } else {
        float x0 = __bfloat162float(rowp[j]), x1 = __bfloat162float(rowp[j + NP]);
        float2 y = rot(x0, x1, c);
        rowp[j] = __float2bfloat16(y.x);
        rowp[j + NP] = __float2bfloat16(y.y);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// v4：一线程一个 uint4（8 个 rope 元素 = 4 个交错 pair），把 rope 区当连续 uint4 流；
//   只对交错配对有效（PAIR=0）。连续线程访问连续 uint4 → 合并。
// ---------------------------------------------------------------------------
template <int R, bool BF16TAB>
__global__ void __launch_bounds__(256) rope_vec4_kernel(
    bf16* __restrict__ X, int T, int H, int HD, const int* __restrict__ pos,
    const void* __restrict__ cs) {
  const int NP = R / 2;
  const int U4 = R / 8;
  const long total = (long)T * H * U4;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const long row = i / U4;
    const int u = (int)(i % U4);
    const long t = row / H;
    const int p = pos ? pos[t] : (int)t;
    bf16* rowp = X + row * (long)HD + (HD - R);
    uint4 v = reinterpret_cast<uint4*>(rowp)[u];
    bf16* vb = reinterpret_cast<bf16*>(&v);
    const int j0 = u * 4;
    float x[8];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      float2 c = read_cs<BF16TAB>(cs, p * NP + j0 + m);
      float a = __bfloat162float(vb[2 * m]), b = __bfloat162float(vb[2 * m + 1]);
      float2 y = rot(a, b, c);
      x[2 * m] = y.x;
      x[2 * m + 1] = y.y;
    }
    uint4 o;
    bf16* ob = reinterpret_cast<bf16*>(&o);
#pragma unroll
    for (int m = 0; m < 8; ++m) ob[m] = __float2bfloat16(x[m]);
    reinterpret_cast<uint4*>(rowp)[u] = o;
  }
}

// ---------------------------------------------------------------------------
// v6：v4 的多 uint4/线程版本（VPT=2/4），提高每线程在飞字节数与 MLP。
//   只对交错配对有效（PAIR=0）。
// ---------------------------------------------------------------------------
template <int R, int VPT, bool BF16TAB>
__global__ void __launch_bounds__(256) rope_vec4_multi_kernel(
    bf16* __restrict__ X, int T, int H, int HD, const int* __restrict__ pos,
    const void* __restrict__ cs) {
  const int NP = R / 2;
  const int U4 = R / 8;
  const long total = (long)T * H * U4;
  const int lane = threadIdx.x & 31;
  (void)lane;
  const long stride = (long)gridDim.x * blockDim.x;
  for (long i0 = ((long)blockIdx.x * blockDim.x + threadIdx.x) * VPT; i0 < total;
       i0 += stride * VPT) {
#pragma unroll
    for (int v = 0; v < VPT; ++v) {
      const long i = i0 + v;
      if (i >= total) break;
      const long row = i / U4;
      const int u = (int)(i % U4);
      const long t = row / H;
      const int p = pos ? pos[t] : (int)t;
      bf16* rowp = X + row * (long)HD + (HD - R);
      uint4 val = reinterpret_cast<uint4*>(rowp)[u];
      bf16* vb = reinterpret_cast<bf16*>(&val);
      const int j0 = u * 4;
      float x[8];
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        float2 c = read_cs<BF16TAB>(cs, p * NP + j0 + m);
        float a = __bfloat162float(vb[2 * m]), b = __bfloat162float(vb[2 * m + 1]);
        float2 y = rot(a, b, c);
        x[2 * m] = y.x;
        x[2 * m + 1] = y.y;
      }
      uint4 o;
      bf16* ob = reinterpret_cast<bf16*>(&o);
#pragma unroll
      for (int m = 0; m < 8; ++m) ob[m] = __float2bfloat16(x[m]);
      reinterpret_cast<uint4*>(rowp)[u] = o;
    }
  }
}

// ---------------------------------------------------------------------------
// v5：一 warp 一 token，cos/sin 常驻寄存器，内层循环摊到 H 个 head
//   同一 token 的 H 个 head 共享同一份 rope 表 —— 表流量 ÷H
// ---------------------------------------------------------------------------
template <int R, int PAIR, bool BF16TAB>
__global__ void __launch_bounds__(256) rope_tokwarp_kernel(
    bf16* __restrict__ X, int T, int H, int HD, const int* __restrict__ pos,
    const void* __restrict__ cs) {
  const int NP = R / 2;
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int nwarp = (gridDim.x * blockDim.x) >> 5;
  for (long t = warp; t < T; t += nwarp) {
    const int p = pos ? pos[t] : (int)t;
    float2 c0 = read_cs<BF16TAB>(cs, p * NP + lane);
    float2 c1 = make_float2(0.f, 0.f);
    if constexpr (NP > 32) c1 = read_cs<BF16TAB>(cs, p * NP + lane + 32);
    bf16* base = X + t * (long)H * HD + (HD - R);
#pragma unroll 4
    for (int h = 0; h < H; ++h) {
      bf16* rowp = base + (long)h * HD;
      if constexpr (PAIR == 0) {
        uint32_t w = *reinterpret_cast<uint32_t*>(rowp + 2 * lane);
        float x0 = __bfloat162float(reinterpret_cast<bf16*>(&w)[0]);
        float x1 = __bfloat162float(reinterpret_cast<bf16*>(&w)[1]);
        float2 y = rot(x0, x1, c0);
        uint32_t o;
        reinterpret_cast<bf16*>(&o)[0] = __float2bfloat16(y.x);
        reinterpret_cast<bf16*>(&o)[1] = __float2bfloat16(y.y);
        *reinterpret_cast<uint32_t*>(rowp + 2 * lane) = o;
      } else {
        // R=128：lane 处理 pair lane 与 lane+32
        float a0 = __bfloat162float(rowp[lane]), a1 = __bfloat162float(rowp[lane + NP]);
        float2 y0 = rot(a0, a1, c0);
        rowp[lane] = __float2bfloat16(y0.x);
        rowp[lane + NP] = __float2bfloat16(y0.y);
        float b0 = __bfloat162float(rowp[lane + 32]), b1 = __bfloat162float(rowp[lane + 32 + NP]);
        float2 y1 = rot(b0, b1, c1);
        rowp[lane + 32] = __float2bfloat16(y1.x);
        rowp[lane + 32 + NP] = __float2bfloat16(y1.y);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// v4s：对半配对 R=128 的向量化 —— 一线程吃 4 个 pair（uint2 各取两半）
// ---------------------------------------------------------------------------
template <int R, bool BF16TAB>
__global__ void __launch_bounds__(256) rope_vec4_split_kernel(
    bf16* __restrict__ X, int T, int H, int HD, const int* __restrict__ pos,
    const void* __restrict__ cs) {
  const int NP = R / 2;      // 64
  const int TPR = R / 8;     // 每行 16 个线程
  const long total = (long)T * H * TPR;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const long row = i / TPR;
    const int tt = (int)(i % TPR);
    const long t = row / H;
    const int p = pos ? pos[t] : (int)t;
    bf16* rowp = X + row * (long)HD + (HD - R);
    const int j0 = tt * 4;
    uint2 a = *reinterpret_cast<uint2*>(rowp + j0);
    uint2 b = *reinterpret_cast<uint2*>(rowp + NP + j0);
    bf16* ab = reinterpret_cast<bf16*>(&a);
    bf16* bb = reinterpret_cast<bf16*>(&b);
    uint2 oa, ob;
    bf16* oab = reinterpret_cast<bf16*>(&oa);
    bf16* obb = reinterpret_cast<bf16*>(&ob);
#pragma unroll
    for (int m = 0; m < 4; ++m) {
      float2 c = read_cs<BF16TAB>(cs, p * NP + j0 + m);
      float x0 = __bfloat162float(ab[m]), x1 = __bfloat162float(bb[m]);
      float2 y = rot(x0, x1, c);
      oab[m] = __float2bfloat16(y.x);
      obb[m] = __float2bfloat16(y.y);
    }
    *reinterpret_cast<uint2*>(rowp + j0) = oa;
    *reinterpret_cast<uint2*>(rowp + NP + j0) = ob;
  }
}

// ---------------------------------------------------------------------------
// 诊断：同访问模式的 copy（读+写，无旋转/表）与只读 roof，用来定位「模式天花板」
// ---------------------------------------------------------------------------
template <int R>
__global__ void rope_copy_kernel(bf16* __restrict__ X, int T, int H, int HD,
                                 unsigned* __restrict__ out) {
  const int U4 = R / 8;
  const long total = (long)T * H * U4;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const long row = i / U4;
    const int u = (int)(i % U4);
    bf16* rowp = X + row * (long)HD + (HD - R);
    uint4* q = reinterpret_cast<uint4*>(rowp);
    uint4 v = q[u];
    v.x ^= 1u;  // 防止编译器把 q[u]=q[u] 优化掉
    q[u] = v;
  }
  (void)out;
}

template <int R>
__global__ void rope_read_kernel(const bf16* __restrict__ X, int T, int H, int HD,
                                 unsigned* __restrict__ out) {
  const int U4 = R / 8;
  const long total = (long)T * H * U4;
  unsigned acc = 0;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += (long)gridDim.x * blockDim.x) {
    const long row = i / U4;
    const int u = (int)(i % U4);
    const bf16* rowp = X + row * (long)HD + (HD - R);
    uint4 v = reinterpret_cast<const uint4*>(rowp)[u];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0xDEADBEEFu) atomicAdd(out, 1u);
}

// ---------------------------------------------------------------------------
// CPU 参考（fp64，复刻 model.py 语义）
// ---------------------------------------------------------------------------
static void rope_ref(const std::vector<bf16>& in, std::vector<bf16>& out, int T, int H, int HD,
                     int R, int PAIR, const int* pos, const std::vector<float>& freq) {
  out = in;
  const int NP = R / 2;
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h) {
      int p = pos ? pos[t] : t;
      bf16* row = out.data() + ((long)t * H + h) * HD + (HD - R);
      for (int j = 0; j < NP; ++j) {
        int i0, i1;
        if (PAIR == 0) { i0 = 2 * j; i1 = 2 * j + 1; } else { i0 = j; i1 = j + NP; }
        double x0 = (double)__bfloat162float(row[i0]), x1 = (double)__bfloat162float(row[i1]);
        double a = (double)p * freq[j];
        double c = cos(a), s = sin(a);
        row[i0] = __float2bfloat16((float)(x0 * c - x1 * s));
        row[i1] = __float2bfloat16((float)(x0 * s + x1 * c));
      }
    }
}

// ---------------------------------------------------------------------------
// 场景
// ---------------------------------------------------------------------------
struct Scenario {
  const char* name;
  int T, H, HD, R;
  int PAIR;  // 0=interleaved(DeepSeek), 1=split-half(Qwen3)
  float base, factor;
  int orig, beta_fast, beta_slow;
};

static Scenario get_scenario(const std::string& n) {
  if (n == "ds-q") return {"ds-mla-q", 16384, 128, 512, 64, 0, 10000.f, 16.f, 65536, 32, 1};
  if (n == "ds-kv") return {"ds-mla-kv", 16384, 1, 512, 64, 0, 10000.f, 16.f, 65536, 32, 1};
  if (n == "qwen-q") return {"qwen3-q", 16384, 40, 128, 128, 1, 1000000.f, 1.f, 0, 32, 1};
  if (n == "qwen-k") return {"qwen3-k", 16384, 8, 128, 128, 1, 1000000.f, 1.f, 0, 32, 1};
  if (n == "ds-q-packed") return {"ds-mla-q-packed", 16384, 128, 64, 64, 0, 10000.f, 16.f, 65536, 32, 1};
  if (n == "ds-q-8k") return {"ds-mla-q-8k", 8192, 128, 512, 64, 0, 10000.f, 16.f, 65536, 32, 1};
  if (n == "ds-q-32k") return {"ds-mla-q-32k", 32768, 128, 512, 64, 0, 10000.f, 16.f, 65536, 32, 1};
  std::fprintf(stderr, "unknown scenario '%s'\n", n.c_str());
  std::exit(1);
}

// ---------------------------------------------------------------------------
// 单场景基准（PAIR/R 编译期）
// ---------------------------------------------------------------------------
template <int R, int PAIR>
static void bench_scenario(const Scenario& sc, bool dump, const DeviceInfo& d) {
  const int NP = R / 2;
  std::vector<float> freq = build_freqs(R, sc.base, sc.orig, sc.factor, sc.beta_fast, sc.beta_slow);
  const int maxpos = sc.T;
  std::vector<float2> cs = build_cs(freq, maxpos);
  std::vector<__nv_bfloat162> csb((size_t)maxpos * NP);
  for (size_t i = 0; i < cs.size(); ++i) csb[i] = __floats2bfloat162_rn(cs[i].x, cs[i].y);

  const long nrow = (long)sc.T * sc.H;
  const long nelem = nrow * sc.HD;
  const size_t bytes = (size_t)nrow * R * sizeof(bf16) * 2;  // 读+写 rope 区

  std::vector<bf16> hX(nelem);
  for (long i = 0; i < nelem; ++i) hX[i] = __float2bfloat16((float)((rand() % 2000 - 1000) / 1000.0));

  bf16* dX; CUDA_CHECK(cudaMalloc(&dX, nelem * sizeof(bf16)));
  unsigned* dOut; CUDA_CHECK(cudaMalloc(&dOut, sizeof(unsigned)));
  float2* dcs; CUDA_CHECK(cudaMalloc(&dcs, cs.size() * sizeof(float2)));
  __nv_bfloat162* dcsb; CUDA_CHECK(cudaMalloc(&dcsb, csb.size() * sizeof(__nv_bfloat162)));
  CUDA_CHECK(cudaMemcpy(dcs, cs.data(), cs.size() * sizeof(float2), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dcsb, csb.data(), csb.size() * sizeof(__nv_bfloat162), cudaMemcpyHostToDevice));

  std::printf("=== %s : T=%d H=%d HD=%d R=%d PAIR=%s base=%g yarn=%d (bytes=%.1f MB) ===\n",
              sc.name, sc.T, sc.H, sc.HD, sc.R, PAIR ? "split-half" : "interleaved", sc.base, sc.orig,
              bytes / 1e6);

  const long nthreads = 256;

  // 正确性
  {
    std::vector<bf16> ref;
    rope_ref(hX, ref, sc.T, sc.H, sc.HD, R, PAIR, nullptr, freq);
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), nelem * sizeof(bf16), cudaMemcpyHostToDevice));
    const int blk = (int)((nrow + nthreads - 1) / nthreads);
    rope_warp_kernel<R, PAIR, false><<<blk, 256>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcs);
    CUDA_CHECK_LAST();
    std::vector<bf16> got(nelem);
    CUDA_CHECK(cudaMemcpy(got.data(), dX, nelem * sizeof(bf16), cudaMemcpyDeviceToHost));
    double maxerr = 0; long checked = 0;
    for (int t = 0; t < sc.T; t += sc.T / 16 + 1)
      for (int h = 0; h < sc.H; ++h) {
        const bf16* a = ref.data() + ((long)t * sc.H + h) * sc.HD + (sc.HD - R);
        const bf16* b = got.data() + ((long)t * sc.H + h) * sc.HD + (sc.HD - R);
        for (int j = 0; j < R; ++j) {
          double e = std::fabs((double)__bfloat162float(a[j]) - (double)__bfloat162float(b[j]));
          maxerr = std::max(maxerr, e);
        }
        checked += R;
      }
    std::printf("  [correctness] warp-v2 vs CPU-fp64: max_abs_err=%.3e over %ld elems  %s\n",
                maxerr, checked, maxerr < 2e-2 ? "OK" : "FAIL");
  }

  if (dump) {
    std::string base = std::string("dump_") + sc.name;
    FILE* f = fopen((base + "_meta.txt").c_str(), "w");
    fprintf(f, "%d %d %d %d %d %g %g %d %d %d\n", sc.T, sc.H, sc.HD, R, PAIR, sc.base, sc.factor,
            sc.orig, sc.beta_fast, sc.beta_slow);
    fclose(f);
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), nelem * sizeof(bf16), cudaMemcpyHostToDevice));
    const int blk = (int)((nrow + nthreads - 1) / nthreads);
    rope_warp_kernel<R, PAIR, false><<<blk, 256>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcs);
    CUDA_CHECK_LAST();
    std::vector<bf16> got(nelem);
    CUDA_CHECK(cudaMemcpy(got.data(), dX, nelem * sizeof(bf16), cudaMemcpyDeviceToHost));
    f = fopen((base + "_in.bin").c_str(), "wb"); fwrite(hX.data(), sizeof(bf16), nelem, f); fclose(f);
    f = fopen((base + "_out.bin").c_str(), "wb"); fwrite(got.data(), sizeof(bf16), nelem, f); fclose(f);
    std::printf("  [dump] wrote %s_{meta.txt,in.bin,out.bin}\n", base.c_str());
  }

  auto restore = [&] { CUDA_CHECK(cudaMemcpy(dX, hX.data(), nelem * sizeof(bf16), cudaMemcpyHostToDevice)); };
  const int blocks = (int)((nrow + nthreads - 1) / nthreads);

#define RUN(name, ...)                                              \
  do {                                                              \
    auto l = [&] { __VA_ARGS__; };                                  \
    restore();                                                      \
    double ms = bench_ms(l, 10, 50);                                \
    report_mem(name, ms, (double)bytes, d);                         \
  } while (0)

  RUN("roof: pattern copy (r+w)",
      rope_copy_kernel<R><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, dOut));
  {
    auto l = [&] { rope_read_kernel<R><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, dOut); };
    restore();
    double ms = bench_ms(l, 10, 50);
    report_mem("roof: pattern read-only", ms, (double)bytes / 2, d);
  }
  if (sc.T <= 16384) {  // v0 现场算 double pow/cos，32k 会跑很久
    RUN("v0 naive (on-the-fly)",
        rope_naive_kernel<R, PAIR><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, sc.base,
                                                              sc.factor, sc.orig, sc.beta_fast, sc.beta_slow));
  }
  RUN("v1a elementwise fp32-tab",
      rope_tab_kernel<R, PAIR, false><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcs));
  RUN("v1b elementwise bf16-tab",
      rope_tab_kernel<R, PAIR, true><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb));
  RUN("v2a warp fp32-tab",
      rope_warp_kernel<R, PAIR, false><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcs));
  RUN("v2b warp bf16-tab",
      rope_warp_kernel<R, PAIR, true><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb));
  if constexpr (PAIR == 0) {
    RUN("v4 vec4 (uint4 stream)",
        rope_vec4_kernel<R, true><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb));
    RUN("v6 vec4 VPT=2",
        rope_vec4_multi_kernel<R, 2, true><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb));
    RUN("v6 vec4 VPT=4",
        rope_vec4_multi_kernel<R, 4, true><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb));
    for (int g : {132, 264, 528, 1056, 2112, 4224}) {
      auto l = [&] { rope_vec4_multi_kernel<R, 2, true><<<g, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb); };
      restore();
      double ms = bench_ms(l, 10, 50);
      char nm[64];
      snprintf(nm, sizeof(nm), "v6 VPT=2 grid=%d", g);
      report_mem(nm, ms, (double)bytes, d);
    }
  }
  if constexpr (PAIR == 1) {
    RUN("v4s split vec4 (uint2x2)",
        rope_vec4_split_kernel<R, true><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb));
  }
  RUN("v5 tok-warp reg-tab",
      rope_tokwarp_kernel<R, PAIR, true><<<blocks, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb));
  for (int g : {132, 264, 528, 1056, 2112}) {
    auto l = [&] { rope_tokwarp_kernel<R, PAIR, true><<<g, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb); };
    restore();
    double ms = bench_ms(l, 10, 50);
    char nm[64];
    snprintf(nm, sizeof(nm), "v5 grid=%d", g);
    report_mem(nm, ms, (double)bytes, d);
  }
  for (int g : {132, 264, 528, 1056, 2112}) {
    auto l = [&] { rope_warp_kernel<R, PAIR, true><<<g, (int)nthreads>>>(dX, sc.T, sc.H, sc.HD, nullptr, dcsb); };
    restore();
    double ms = bench_ms(l, 10, 50);
    char nm[64];
    snprintf(nm, sizeof(nm), "v2b grid=%d", g);
    report_mem(nm, ms, (double)bytes, d);
  }
#undef RUN

  cudaFree(dX); cudaFree(dcs); cudaFree(dcsb); cudaFree(dOut);
  std::printf("\n");
}

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "bench";
  const std::string which = argc > 2 ? argv[2] : "all";
  const bool dump = (mode == "dump");

  DeviceInfo d = device_info(0);
  print_device_info(d);
  std::printf("\n");

  std::vector<std::string> names;
  if (which == "all") names = {"ds-q", "ds-kv", "qwen-q", "qwen-k", "ds-q-8k", "ds-q-32k"};
  else names = {which};

  for (const auto& sn : names) {
    Scenario sc = get_scenario(sn);
    if (sc.R == 64 && sc.PAIR == 0) bench_scenario<64, 0>(sc, dump, d);
    else if (sc.R == 128 && sc.PAIR == 1) bench_scenario<128, 1>(sc, dump, d);
    else if (sc.R == 64 && sc.PAIR == 1) bench_scenario<64, 1>(sc, dump, d);
    else if (sc.R == 128 && sc.PAIR == 0) bench_scenario<128, 0>(sc, dump, d);
    else { std::fprintf(stderr, "no instantiation for R=%d PAIR=%d\n", sc.R, sc.PAIR); }
  }
  return 0;
}
