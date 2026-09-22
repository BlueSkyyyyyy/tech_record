// 最小单 tile GEMM：TMA A + 手写 SW128 B（MODE0）与 TMA B（MODE2）对比 CPU。
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//         scripts/run.sh 57-v4-fp4-moe-fold/gemm_smoke.cu
#include "../common/cuda_utils.cuh"
#include <cuda.h>
#include <cuda_fp8.h>

constexpr int BM = 128, BN = 128, BK = 128;
constexpr int KTOT = 7168;
using fp8 = __nv_fp8_e4m3;

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_m64n128k32(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, %67, %68;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
__device__ __forceinline__ uint32_t k32_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ int sw128_off(int r, int k) {
  const int rr = r & 7;
  return (r >> 3) * 1024 + (rr * 8 + (((k >> 4) & 7) ^ rr)) * 16 + (k & 15);
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tm, void* dst, int c0, int c1, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(dst)),
               "l"(reinterpret_cast<uint64_t>(tm)), "r"(smem_u32(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(c));
}
__device__ __forceinline__ void mbar_aet(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(n));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t p) {
  asm volatile("{\n.reg .pred q;\nW%=: mbarrier.try_wait.parity.shared::cta.b64 q, [%0], %1;\n@!q bra W%=;\n}\n" ::"r"(smem_u32(b)), "r"(p));
}

// MODE 0 = 手写 B 布局；MODE 1 = TMA B
template <int MODE>
__global__ void __launch_bounds__(256) smoke(const __grid_constant__ CUtensorMap tmA,
                                             const __grid_constant__ CUtensorMap tmB,
                                             const uint8_t* __restrict__ Bglob,
                                             float* __restrict__ D) {
  extern __shared__ __align__(1024) char smem[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
  char* As = smem + 1024;
  char* Bs = As + BM * BK;
  char* Braw = Bs + BN * BK;
  const int tid = threadIdx.x;
  if (tid == 0) { mbar_init(bar, 1); }
  __syncthreads();
  if (tid == 0) {
    mbar_aet(bar, BM * BK + (MODE == 1 ? BN * BK : 0));
    tma_load_2d(&tmA, As, 0, 0, bar);
    if (MODE == 1) tma_load_2d(&tmB, Bs, 0, 0, bar);
  }
  if (MODE == 0) {
    // cp.async B 到 Braw，再手写 SW128
    if (tid < 32) {
      for (int i = tid; i < BN * BK / 16; i += 32) {
        const int rr = i / (BK / 16), rc = i % (BK / 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(Braw + i * 16)),
                     "l"(Bglob + (size_t)rr * KTOT + rc * 16));
      }
      asm volatile("cp.async.commit_group;");
    }
    __syncthreads();
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    for (int i = tid; i < BN * BK; i += blockDim.x) {
      const int n = i / BK, k = i % BK;
      Bs[sw128_off(n, k)] = reinterpret_cast<const uint8_t*>(Braw)[i];
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  mbar_wait(bar, 0);
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5;
  const int wg = tid >> 7;
  const int row0 = wg * 64 + (warp & 3) * 16 + (lane >> 2);
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.f;
  wgmma_fence();
#pragma unroll
  for (int s = 0; s < BK / 32; ++s) {
    uint64_t da = make_desc_sw128(k32_addr(smem_u32(As) + wg * 64 * BK, s), 1024);
    uint64_t db = make_desc_sw128(k32_addr(smem_u32(Bs), s), 1024);
    wgmma_m64n128k32(acc, da, db);
  }
  wgmma_commit();
  wgmma_wait0();
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int col = j * 8 + (lane & 3) * 2;
    D[(size_t)row0 * BN + col] = acc[j * 4 + 0];
    D[(size_t)row0 * BN + col + 1] = acc[j * 4 + 1];
    D[(size_t)(row0 + 8) * BN + col] = acc[j * 4 + 2];
    D[(size_t)(row0 + 8) * BN + col + 1] = acc[j * 4 + 3];
  }
}

__host__ __device__ float fp8f(unsigned char b) {
  __nv_fp8_e4m3 v; *reinterpret_cast<unsigned char*>(&v) = b; return (float)v;
}
static CUtensorMap mk(uint8_t* p, int R, int K, int boxR, int boxK) {
  CUtensorMap tm; cuuint64_t d[2] = {(cuuint64_t)K, (cuuint64_t)R}; cuuint64_t s[1] = {(cuuint64_t)K};
  cuuint32_t b[2] = {(cuuint32_t)boxK, (cuuint32_t)boxR}; cuuint32_t e[2] = {1, 1};
  cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, p, d, s, b, e, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return tm;
}
int main() {
  DeviceInfo di = device_info(0); print_device_info(di);
  uint8_t *hA = new uint8_t[BM * KTOT], *hB = new uint8_t[BN * KTOT];
  for (int i = 0; i < BM * KTOT; ++i) { float f = ((i * 37) % 17 - 8) * 0.25f; fp8 t = __nv_fp8_e4m3(f); hA[i] = *reinterpret_cast<uint8_t*>(&t); }
  for (int i = 0; i < BN * KTOT; ++i) { float f = ((i * 53) % 13 - 6) * 0.5f; fp8 t = __nv_fp8_e4m3(f); hB[i] = *reinterpret_cast<uint8_t*>(&t); }
  // CPU ref
  std::vector<float> ref(BM * BN, 0.f);
  for (int m = 0; m < BM; ++m)
    for (int n = 0; n < BN; ++n) { double s = 0; for (int k = 0; k < BK; ++k) s += (double)fp8f(hA[m * KTOT + k]) * fp8f(hB[n * KTOT + k]); ref[m * BN + n] = (float)s; }
  uint8_t *dA, *dB; float* dD;
  CUDA_CHECK(cudaMalloc(&dA, BM * KTOT)); CUDA_CHECK(cudaMalloc(&dB, BN * KTOT)); CUDA_CHECK(cudaMalloc(&dD, BM * BN * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA, BM * KTOT, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, BN * KTOT, cudaMemcpyHostToDevice));
  CUtensorMap tmA = mk(dA, BM, KTOT, BM, BK);
  CUtensorMap tmB = mk(dB, BN, KTOT, BN, BK);
  size_t shm = 1024 + BM * BK + BN * BK + BN * BK + 4096;
  for (int mode = 0; mode < 2; ++mode) {
    CUDA_CHECK(cudaMemset(dD, 0, BM * BN * 4));
    if (mode == 0) {
      auto fn = smoke<0>; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
      fn<<<1, 256, shm>>>(tmA, tmB, dB, dD);
    } else {
      auto fn = smoke<1>; cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
      fn<<<1, 256, shm>>>(tmA, tmB, dB, dD);
    }
    CUDA_CHECK_LAST(); CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> got(BM * BN);
    CUDA_CHECK(cudaMemcpy(got.data(), dD, BM * BN * 4, cudaMemcpyDeviceToHost));
    double mx = 0; int bad = 0;
    for (int i = 0; i < BM * BN; ++i) { double e = std::fabs(got[i] - ref[i]); mx = std::max(mx, e); if (e > 1.0) bad++; }
    printf("MODE%d: max_abs=%.3e bad=%d  sample got[0]=%.2f ref[0]=%.2f got[127]=%.2f ref[127]=%.2f  %s\n",
           mode, mx, bad, got[0], ref[0], got[127], ref[127], bad == 0 ? "OK" : "FAIL");
  }
  return 0;
}
