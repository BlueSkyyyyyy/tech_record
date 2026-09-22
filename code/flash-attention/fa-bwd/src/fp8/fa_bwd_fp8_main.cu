// =============================================================================
// fa_bwd_fp8_main.cu —— FlashAttention 反向（FP8）**两文件版的 host 部分**
// =============================================================================
// 由 P3-4 单文件 `fa_bwd_fp8_mma_onefile.cu` 拆分而来（P3-5）：device 代码移入
// `fa_bwd_fp8_kernels.cuh`，本文件只保留 host 侧：npy 读取 / launcher / 自测对拍。
// 行为与单文件版**逐指标一致**（kernel 代码逐字未改）。
//
// 用法：run.sh src/fp8/fa_bwd_fp8_main.cu [--dir=...] [--full|--causal] [--o=...] [--iters=N]
// =============================================================================

#include "fa_bwd_fp8_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// =============================================================================
// 极简 npy 读取（little-endian C-contiguous float32）
// =============================================================================
struct NpyF32 {
  std::vector<float> data;
  std::vector<long> shape;
};

static NpyF32 load_npy_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "无法打开 %s\n", path.c_str());
    std::exit(1);
  }
  char magic[6];
  f.read(magic, 6);
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h16 = 0;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hlen = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }
  std::string header(hlen, '\0');
  f.read(&header[0], hlen);
  if (header.find("f4") == std::string::npos) {
    fprintf(stderr, "%s 不是 float32 npy\n", path.c_str());
    std::exit(1);
  }
  NpyF32 out;
  size_t sp = header.find("'shape'");
  size_t lp = header.find('(', sp);
  size_t rp = header.find(')', lp);
  std::string tup = header.substr(lp + 1, rp - lp - 1);
  for (size_t i = 0; i < tup.size();) {
    while (i < tup.size() && (tup[i] == ' ' || tup[i] == ',')) ++i;
    long v = 0;
    bool any = false;
    while (i < tup.size() && tup[i] >= '0' && tup[i] <= '9') {
      v = v * 10 + (tup[i] - '0');
      ++i;
      any = true;
    }
    if (any) out.shape.push_back(v);
  }
  std::streampos pos = f.tellg();
  f.seekg(0, std::ios::end);
  size_t total = (size_t)f.tellg();
  f.seekg(pos);
  size_t nbytes = total - (size_t)pos;
  out.data.resize(nbytes / sizeof(float));
  f.read(reinterpret_cast<char*>(out.data.data()), nbytes);
  return out;
}

struct DiffStat {
  double max_abs;
  double max_rel;
};

static DiffStat diff_stat(const std::vector<float>& a, const std::vector<float>& b) {
  DiffStat st{0.0, 0.0};
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    if (d > st.max_abs) st.max_abs = d;
    double r = d / (std::fabs((double)b[i]) + 1e-3);
    if (r > st.max_rel) st.max_rel = r;
  }
  return st;
}

// =============================================================================
// host / launcher / self-test
// =============================================================================
int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8";
  std::string o_name = "ref_o";
  bool causal = true;
  int iters = 20;
  int ksplit = -1;  // -1 = 自动
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--full") causal = false;
    else if (a == "--causal") causal = true;
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--ksplit=", 0) == 0) ksplit = atoi(a.c_str() + 9);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  auto q_np = load_npy_f32(dir + "/q.npy");
  auto k_np = load_npy_f32(dir + "/k.npy");
  auto v_np = load_npy_f32(dir + "/v.npy");
  auto do_np = load_npy_f32(dir + "/do.npy");
  auto o_np = load_npy_f32(dir + "/" + o_name + ".npy");
  auto rdq = load_npy_f32(dir + "/ref_dq.npy");
  auto rdk = load_npy_f32(dir + "/ref_dk.npy");
  auto rdv = load_npy_f32(dir + "/ref_dv.npy");

  if (q_np.shape.size() != 4) {
    fprintf(stderr, "期望 q 为 4D [B,S,H,D]\n");
    return 1;
  }
  const int B = (int)q_np.shape[0], S = (int)q_np.shape[1];
  const int H = (int)q_np.shape[2], D = (int)q_np.shape[3];
  if (D != kHeadDim) {
    fprintf(stderr, "本版本仅支持 head_dim=%d（当前 %d）\n", kHeadDim, D);
    return 1;
  }
  const size_t n = (size_t)B * S * H * D;
  const size_t rows = (size_t)B * S * H;
  const float scale = 1.0f / sqrtf((float)D);

  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d D=%d causal=%d scale=%.6f\n", B, S, H, D, (int)causal, scale);
  printf("FP8 mma: Q/K/V=E4M3, dO=E5M2, dS2/dS3=E5M2, Ap=E4M3 (rowwise); P/dS fp32\n");
  printf("smem = %d bytes (%.1f KB); lse smem = %d bytes (%.1f KB)\n", kSmemBytes,
         kSmemBytes / 1024.0, kLseSmemBytes, kLseSmemBytes / 1024.0);

  float *d_q_f, *d_k_f, *d_v_f, *d_do_f, *d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc, *d_dq, *d_dk, *d_dv;
  CUDA_CHECK(cudaMalloc(&d_q_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_k_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_v_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_do_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_o_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_q8, n));
  CUDA_CHECK(cudaMalloc(&d_k8, n));
  CUDA_CHECK(cudaMalloc(&d_v8, n));
  CUDA_CHECK(cudaMalloc(&d_do8, n));
  CUDA_CHECK(cudaMalloc(&d_qs, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_ks, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_vs, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_dos, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_delta, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_lse, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dq, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dk, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dv, n * 4));

  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), n * 4, cudaMemcpyHostToDevice));

  auto quant = [&]() {
    quantize_row_kernel<<<(int)rows, 128>>>(d_q_f, d_q8, d_qs, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, 128>>>(d_k_f, d_k8, d_ks, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, 128>>>(d_v_f, d_v8, d_vs, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, 128>>>(d_do_f, d_do8, d_dos, kHeadDim, 1);
  };

  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_mma_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBytes));
  // ---- O2b：自动选择 N 方向切块数。base = 未切块时的 CTA 数；目标是让 grid 至少铺满
  //      一个波（132 SM × 3 CTA/SM ≈ 396 个并发槽），小 S 时把空转的 SM 用起来。----
  const long base_grid = (long)((S + BM - 1) / BM) * H * B;
  if (ksplit < 1) {
    const long wave_slots = 132L * 3L;
    long k = (base_grid + wave_slots - 1) / base_grid;  // 向上取整
    if (k < 1) k = 1;
    if (k > 4) k = 4;
    ksplit = (int)k;
  }
  dim3 pg(S, H, B);
  dim3 lg((S + LBM - 1) / LBM, H, B);
  dim3 mg((S + BM - 1) / BM * ksplit, H, B);
  printf("grid main = %d x %d x %d  (ksplit=%d, base_grid=%ld)\n", mg.x, mg.y, mg.z,
         ksplit, base_grid);
  const int cvt_threads = 256;
  const int cvt_blocks = (int)std::min<size_t>((n + cvt_threads - 1) / cvt_threads, 65535);

  auto run_preprocess = [&]() {
    lse_mma_kernel<<<lg, THREADS, kLseSmemBytes>>>(d_q8, d_qs, d_k8, d_ks, d_lse, S, H,
                                                   scale, (int)causal);
    delta_kernel<<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
  };

  auto run_all = [&]() {
    quant();
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, n * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, n * 4));
    run_preprocess();
    fa_bwd_fp8_mma_kernel<<<mg, THREADS, kSmemBytes>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
        d_dk_acc, d_dv_acc, S, H, scale, (int)causal, ksplit);
    convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                d_dv, n);
  };

  for (int i = 0; i < 3; ++i) run_all();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_all();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  ms /= iters;
  double flops = 4.0 * (double)B * S * H * S * D;
  printf("[timing] total(quant+pre+main+cvt) %.4f ms  %.2f TFLOPS (bwd FLOPs=4BS^2HD)\n", ms,
         flops / (ms * 1e-3) / 1e12);

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) quant();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_quant = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_quant, ev0, ev1));
  ms_quant /= iters;

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_preprocess();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_pre = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev0, ev1));
  ms_pre /= iters;

  CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * 4));
  CUDA_CHECK(cudaMemset(d_dk_acc, 0, n * 4));
  CUDA_CHECK(cudaMemset(d_dv_acc, 0, n * 4));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i)
    fa_bwd_fp8_mma_kernel<<<mg, THREADS, kSmemBytes>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
        d_dk_acc, d_dv_acc, S, H, scale, (int)causal, ksplit);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_main = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_main, ev0, ev1));
  ms_main /= iters;
  printf("[timing] quant %.4f ms | preprocess %.4f ms | main %.4f ms | convert %.4f ms\n",
         ms_quant, ms_pre, ms_main, ms - ms_quant - ms_pre - ms_main);

  std::vector<float> mdq(n), mdk(n), mdv(n);
  CUDA_CHECK(cudaMemcpy(mdq.data(), d_dq, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdk.data(), d_dk, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdv.data(), d_dv, n * 4, cudaMemcpyDeviceToHost));

  auto print_cmp = [&](const char* name, const std::vector<float>& mine,
                       const std::vector<float>& ref) {
    DiffStat st = diff_stat(mine, ref);
    printf("  %-4s vs ref: max_abs=%.3e  max_rel=%.3e\n", name, st.max_abs, st.max_rel);
  };
  printf("[compare] ours vs fp32 ref (O from %s.npy)\n", o_name.c_str());
  print_cmp("dq", mdq, rdq.data);
  print_cmp("dk", mdk, rdk.data);
  print_cmp("dv", mdv, rdv.data);

  auto cmp_to = [&](const char* name, const std::vector<float>& mine, const char* fname) {
    std::ifstream test(dir + "/" + fname + ".npy");
    if (!test.good()) return;
    auto t = load_npy_f32(dir + "/" + fname + ".npy");
    DiffStat st = diff_stat(mine, t.data);
    printf("  %-4s vs %s: max_abs=%.3e  max_rel=%.3e\n", name, fname, st.max_abs,
           st.max_rel);
  };
  printf("[compare] ours vs TE FP8\n");
  cmp_to("dq", mdq, "te_dq");
  cmp_to("dk", mdk, "te_dk");
  cmp_to("dv", mdv, "te_dv");

  cudaFree(d_q_f); cudaFree(d_k_f); cudaFree(d_v_f); cudaFree(d_do_f); cudaFree(d_o_f);
  cudaFree(d_q8); cudaFree(d_k8); cudaFree(d_v8); cudaFree(d_do8);
  cudaFree(d_qs); cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_dos);
  cudaFree(d_delta); cudaFree(d_lse);
  cudaFree(d_dq_acc); cudaFree(d_dk_acc); cudaFree(d_dv_acc);
  cudaFree(d_dq); cudaFree(d_dk); cudaFree(d_dv);
  return 0;
}
