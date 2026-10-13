// P150-1 (docs/64): host-side arithmetic probe of the *in-kernel tile
// scheduler* of MPK's SM100 block-scaled FP8 linear family
// (linear_fp8_sm100_scheduler.cuh), and of its DeepGEMM ancestor
// (deep_gemm/scheduler/gemm.cuh).
//
// The scheduler is what a persistent-kernel "one task = one CTA" body uses to
// walk its own output tiles: DeepGEMM's multi-SM grouped-GEMM scheduler was
// copied into MPK and edited for the persistent-kernel launch convention. Every
// method is __device__, but the *arithmetic* is pure combinatorics, so it can
// be re-implemented verbatim on the host and checked (bijection / coverage).
//
// No SM100 hardware is needed (and the target H100/sm90 has none). The probe
// answers four questions:
//   A. get_num_1d_blocks_per_group() tie-break (8 vs 16) on realistic shapes.
//   B. is get_swizzled_block_idx() a bijection over [0,num_blocks)?
//   C. does MPK's get_next_block() (block_offset hard-coded to +0) cover every
//      tile exactly once?  (kNumSMs = 1 -> sequential)
//   D. what would the task_register HACK comment's proposal (kNumSMs =
//      MPK_MAX_NUM_WORKERS, offset +0) actually cover?  vs DeepGEMM's real
//      multi-SM sweep (offset = blockIdx.x, kNumSMs = #SM).
//
// Reconstructed from the pinned snapshot f9eb70c; see docs/64 for line cites.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <set>
#include <algorithm>

using u32 = uint32_t;

static void hdr(const char *s) { printf("\n=== %s ===\n", s); }

// ---- get_num_1d_blocks_per_group (MPK _scheduler.cuh:14-34) ----------------
static u32 num_1d_blocks_per_group(u32 BM, u32 BN, u32 kNumSMs, bool mcastA) {
  u32 best = 0, min_usage = 0xffffffffu;
  for (u32 candidate : {8u, 16u}) {
    u32 usage = mcastA
                    ? candidate * BN + ((kNumSMs + candidate - 1) / candidate) * BM
                    : candidate * BM + ((kNumSMs + candidate - 1) / candidate) * BN;
    if (usage < min_usage) { min_usage = usage; best = candidate; }
  }
  return best;
}

// ---- get_swizzled_block_idx (MPK _scheduler.cuh:105-129) --------------------
// MPK keeps only the finall switch; the SM90-only "@CUDA_ARCH__ < 1000 multicast
// fix" from DeepGEMM (gemm.cuh:122-131) was dropped.
static void swizzle_mpk(u32 block_idx, u32 num_m, u32 num_n, u32 n1d,
                        bool mcastA, u32 &mi, u32 &ni, u32 &num_in_group) {
  u32 primary   = mcastA ? num_n : num_m;
  u32 secondary = mcastA ? num_m : num_n;
  u32 npg   = secondary * n1d;
  u32 g     = block_idx / npg;
  u32 first = g * n1d;
  u32 in    = block_idx % npg;
  num_in_group = std::min(n1d, primary - first);
  if (mcastA) { mi = in / num_in_group; ni = first + in % num_in_group; }
  else        { mi = first + in % num_in_group; ni = in / num_in_group; }
}

// ---- MPK get_next_block (MPK _scheduler.cuh:168-196, Normal branch) ---------
//   next_block_idx = (++current_iter) * kNumSMs + 0        // offset hard-coded
static std::vector<std::pair<u32,u32>> mps_sweep(u32 num_m, u32 num_n, u32 n1d,
                                                 bool mcastA, u32 kNumSMs) {
  u32 num_blocks = num_m * num_n;
  std::vector<std::pair<u32,u32>> out;
  long cur = -1;
  for (;;) {
    long next = (long)(++cur) * (long)kNumSMs + 0;
    if (next >= (long)num_blocks) break;
    u32 mi, ni, ng;
    swizzle_mpk((u32)next, num_m, num_n, n1d, mcastA, mi, ni, ng);
    out.push_back({mi, ni});
  }
  return out;
}

// ---- DeepGEMM multi-SM sweep (gemm.cuh:185-269, Normal branch) --------------
//   next_block_idx = (++current_iter) * kNumSMs + blockIdx.x
// union over blockIdx.x in [0,kNumSMs)
static std::vector<std::pair<u32,u32>> deepgemm_sweep(u32 num_m, u32 num_n,
                                                      u32 n1d, bool mcastA,
                                                      u32 kNumSMs) {
  u32 num_blocks = num_m * num_n;
  std::vector<std::pair<u32,u32>> out;
  for (u32 bid = 0; bid < kNumSMs; ++bid) {
    long cur = -1;
    for (;;) {
      long next = (long)(++cur) * (long)kNumSMs + (long)bid;
      if (next >= (long)num_blocks) break;
      u32 mi, ni, ng;
      swizzle_mpk((u32)next, num_m, num_n, n1d, mcastA, mi, ni, ng);
      out.push_back({mi, ni});
    }
  }
  return out;
}

static void check_bijection(u32 num_m, u32 num_n, u32 n1d, bool mcastA,
                            const char *tag) {
  u32 num_blocks = num_m * num_n;
  std::set<std::pair<u32,u32>> seen;
  int oob = 0;
  for (u32 b = 0; b < num_blocks; ++b) {
    u32 mi, ni, ng;
    swizzle_mpk(b, num_m, num_n, n1d, mcastA, mi, ni, ng);
    if (mi >= num_m || ni >= num_n) oob++;
    seen.insert({mi, ni});
  }
  printf("  %-34s tiles=%5u unique=%5u oob=%d  %s\n", tag, num_blocks,
         (u32)seen.size(), oob,
         (seen.size() == num_blocks && oob == 0) ? "BIJECTION OK" : "MISMATCH");
}

static void report_sweep(const char *what, const char *tag,
                         const std::vector<std::pair<u32,u32>> &v,
                         u32 num_blocks) {
  std::set<std::pair<u32,u32>> seen(v.begin(), v.end());
  printf("  %-28s %-26s visits=%5zu unique=%5zu / %5u  %s\n", what, tag,
         v.size(), seen.size(), num_blocks,
         (v.size() == num_blocks && seen.size() == num_blocks) ? "COVERS ALL ONCE"
                                                               : "GAP/DUP");
}

int main() {
  // Production instantiation (task_register.cc:4815-4868):
  //   BLOCK_M=BLOCK_N=128, kNumMulticast=1, kIsMulticastOnA=false,
  //   kNumSMs=1, kNumGroups=1, GemmType::Normal.
  const u32 BM = 128, BN = 128;

  hdr("(A) get_num_1d_blocks_per_group: 8 vs 16 tie-break");
  printf("  %-8s %-8s %-8s %-8s | chosen  usage(8) usage(16)\n",
         "BM", "BN", "kNumSMs", "mcastA");
  const std::pair<u32,bool> cfgs[] = {{1, false}, {132, false}, {1, true},
                                      {132, true}, {64, false}, {16, false}};
  for (auto const &cfg : cfgs) {
    const u32 nSM = cfg.first;
    const bool mcastA = cfg.second;
    auto usage = [&](u32 c) {
      return mcastA ? c * BN + ((nSM + c - 1) / c) * BM
                    : c * BM + ((nSM + c - 1) / c) * BN;
    };
    printf("  %-8u %-8u %-8u %-8d |   %2u      %5u   %5u\n", BM, BN, nSM,
           (int)mcastA, num_1d_blocks_per_group(BM, BN, nSM, mcastA),
           usage(8), usage(16));
  }
  printf("  -> production (kNumSMs=1): usage(8)=%u ties/beats usage(16)=%u, so 8.\n",
         ((1 + 7) / 8) * BM + 8 * BM, ((1 + 15) / 16) * BM + 16 * BM);

  hdr("(B) get_swizzled_block_idx bijection (MPK form, no SM90 multicast fix)");
  // Production n1d = 8 (from A). Realistic M/N block counts incl. unaligned.
  const u32 n1d = 8;
  check_bijection(32, 32, n1d, false, "M=32 N=32 blocks (4096x4096)");
  check_bijection(13,  5, n1d, false, "M=13 N=5 blocks (unaligned)");
  check_bijection( 1,  9, n1d, false, "M=1  N=9  (thin M)");
  check_bijection(17,  3, n1d, true,  "M=17 N=3  (mcastA on N)");
  check_bijection( 9,  9, n1d, false, "M=9  N=9  (small square)");

  hdr("(C) MPK get_next_block coverage (kNumSMs=1, offset hard-coded +0)");
  for (auto [M, N] : {std::pair<u32,u32>{32, 32}, {13, 5}, {1, 9}, {7, 3}}) {
    u32 nb = M * N;
    char tag[64];
    snprintf(tag, sizeof(tag), "M=%u N=%u (num_blocks=%u)", M, N, nb);
    report_sweep("MPK kNumSMs=1", tag, mps_sweep(M, N, n1d, false, 1), nb);
  }

  hdr("(D) the HACK comment's proposal vs DeepGEMM's real multi-SM sweep");
  printf("  task_register.cc:4815-4825 comment says kNumSMs = "
         "MPK_MAX_NUM_WORKERS\n");
  printf("  (a macro that appears NOWHERE else in f9eb70c), but the emitted "
         "code passes kNumSMs = 1.\n\n");
  const u32 W = 1056;   // an 8-blocks/SM-sized worker grid on a 132-SM card
  for (auto [M, N] : {std::pair<u32,u32>{32, 32}, {8, 8}, {2, 2}}) {
    u32 nb = M * N;
    char tag[64];
    snprintf(tag, sizeof(tag), "M=%u N=%u (num_blocks=%u)", M, N, nb);
    report_sweep("kNumSMs=1 (as shipped)", tag, mps_sweep(M, N, n1d, false, 1), nb);
    char tag2[64];
    snprintf(tag2, sizeof(tag2), "kNumSMs=W=%u (comment)", W);
    report_sweep("", tag2, mps_sweep(M, N, n1d, false, W), nb);
    char tag3[64];
    snprintf(tag3, sizeof(tag3), "DeepGEMM kNumSMs=132");
    report_sweep("", tag3, deepgemm_sweep(M, N, n1d, false, 132), nb);
  }

  hdr("(E) DeepGEMM multi-SM sweep: partition, not overlap");
  for (auto [M, N] : {std::pair<u32,u32>{32, 32}, {13, 5}, {17, 3}}) {
    u32 nb = M * N;
    char tag[64];
    snprintf(tag, sizeof(tag), "M=%u N=%u blocks=%u, #SM=132", M, N, nb);
    report_sweep("DeepGEMM blockIdx.x union", tag,
                 deepgemm_sweep(M, N, n1d, false, 132), nb);
  }

  printf("\nAll checks are arithmetic (no kernel launched); see docs/64.\n");
  return 0;
}
