//===- system_barrier.hpp ----------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
// ---------------------------------------------------------------------------
// 【上游】bytedance/flux（comet 分支，commit 7f98c8a），路径 include/flux/cuda/system_barrier.hpp
// 【一句话】Flux 里所有「跨卡 flag 协议」的底座：flag 就是显存里的一个 int，谁都能原子加、谁都能读；
//           这个文件只定义两件事 —— 读写它们时用什么内存序，以及「怎么等」。
// 本注释只讲三件事：
//   1) 四个原语 arrive_inc / wait_eq / wait_lt / wait_eq_reset 的语义、适用面、失效方式；
//   2) 内联 PTX 为什么必须抬到 sys scope（ld.global.acquire.sys、fence.acq_rel.sys、red.relaxed.sys），
//      gpu scope 会坏在哪一步；
//   3) 「thread 0 自旋 + Sync::sync() 广播」这个形状换来了什么，代价是什么。
// 阅读顺序：ld_acquire / red_release（内存序）→ wait_eq / wait_lt（等什么）→ arrive_inc_get /
//           wait_eq_reset（谁负责清零）→ 最后看 CustomizedGenericBarrier 这个 gpu scope 对照组。
// 术语：acquire（获取）/ release（释放）是 C++ 内存序语义；scope 是「一致性域」，
//       gpu = 单卡，sys = 整机（所有 GPU + host）。
// ---------------------------------------------------------------------------
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/barrier.h"
// 【为什么继承了还要整套重写】基类 cutlass::detail::GenericBarrier<Sync> 已经有 wait_eq / arrive_inc 等，
// 但它内部的 ld_acquire 是 ld.global.acquire.gpu、red_release 是 fence + red.relaxed.gpu —— 一致性域只有本卡。
// 本文件让 GenericSystemBarrier 继承它，再在派生类里重新声明同名的 ld_acquire / red_release，
// 靠名字遮蔽把基类那两条换掉，并把四个原语整套重写。原因很现实：基类这些函数不是虚函数，
// 只换 ld 不换 wait 会得到「一半 gpu scope、一半 sys scope」的混合协议 —— 单机 8 卡多半不复现，
// 跨节点必炸，而且表现为偶发挂死而不是报错。
// 【一句话】这个文件干的事，就是把 flag 协议的一致性域从 gpu 抬到 sys。
namespace cutlass {

namespace detail {
/////////////////////////////////////////////////////////////////////////////////////////////////

// 【Sync 模板参数 = 广播原语】自旋只发生在 thread 0，其余线程必须被某种 barrier 挂住，直到 thread 0 等到；
// 这个「挂住 + 广播」的动作就是 Sync::sync()。
//   SyncthreadsSync → __syncthreads()：整个 CTA 一起等，GEMM mainloop 里的常规选择。
//   SingleThreadSync → 空实现：调用方保证只有单个线程在用这个 barrier（warp 级、单线程 producer）。
// 选错的后果不是慢而是错：CTA 级 barrier 配 SingleThreadSync，其余线程会直接冲过去读还没到的数据。
struct SingleThreadSync {
  CUTLASS_DEVICE
  static void
  sync() {}
};

template <class Sync>
struct GenericSystemBarrier : public GenericBarrier<Sync> {
 protected:
// 【ld_acquire：读 flag 的唯一正确方式】
//   ld.global.acquire.sys.b32 —— acquire 语义：这条 load 之后的访存不能被编译器或硬件提前到它之前；
//   sys scope：读到的是整机一致的那份值，而不只是本卡 L2 的视角。
// 为什么普通 load / gpu scope 不行：flag 的写者是别的 GPU（甚至别的节点上的 GPU），
//   写穿过 NVLink/PCIe 落到本卡显存，这次写不在本卡 gpu scope 的一致性域里 ——
//   它什么时候可见没有任何保证，表现可能是偶发读到旧值，也可能是 spin 到超时。
// 代价：sys scope 的 acquire 每次都要走到比 L1 更远的地方（L2 / 对端），一次自旋就是一次全局往返；
//   这正是下面「只让 thread 0 自旋」的原因：把这份代价从 128 个线程压到 1 个线程。
  /// Load flag, as a strong acquire operation (int specialization)
  CUTLASS_DEVICE
  static int
  ld_acquire(int *ptr) {
    int state = 0;
    // Acquire pattern using acquire modifier
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
    return state;
  }

// 【red_release：发信号 = 先把数据 release 出去，再做一次不取返回值的原子加】
// 两步，顺序不能换：
//   1) fence.acq_rel.sys：把本 CTA 在同步点之前写出的数据刷到 sys 域；
//   2) red.relaxed.sys.global.add：只加不读，fire-and-forget，不必等原子操作的响应回来。
// 为什么不是「一条 release 的原子」：release 语义只覆盖本线程之前的写，而这里要发布的是
//   __syncthreads 之前由 CTA 里所有线程写出的数据。fence 本身只排本线程的序，但前面那条
//   __syncthreads 已经让其它线程的写 happens-before 到 thread 0，fence 再把它们一起抬到 sys 域；
//   随后的 red 用 relaxed 就够了。
//   上游那句 "The fence also releases data that was weakly-written by other threads prior to the
//   last syncthreads" 说的就是这件事。
// 为什么用 red 而不是 atomicAdd_system：不需要旧值 → 不用等 response，寄存器也不必挂着等；
//   在「每个 tile 都要发一次信号」的密度下，这点延迟是能量出来的。
// 去掉 fence 的后果：peer 可能先看到计数 +1、再看到（或永远看不到）数据 —— 读到半成品部分和，静默算错。
  /// Reduce into flag, with release pattern (int specialization)
  CUTLASS_DEVICE
  static void
  red_release(int *ptr, int val) {
    // Release pattern using acq_rel fence + relaxed modifier.  (The fence also releases data
    // that was weakly-written by other threads prior to the last syncthreads)
    asm volatile("fence.acq_rel.sys;\n");
    asm volatile("red.relaxed.sys.global.add.s32 [%0], %1;\n" : : "l"(ptr), "r"(val));
  }

 public:
// 【check_value：只 peek，不等待】给「先看一眼，没到就去干别的」的场景用。
// 注意它只返回当前值，不提供任何「等到某个值」的保证，所以调用方必须自己处理「还没到」。
  /// Uses thread[0] to wait for at least the specified count of signals on the given flag counter
  CUTLASS_DEVICE
  static int
  check_value(void *lock_ptr, int thread_idx, int flag_idx) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    return ld_acquire(flag_ptr);
  }

// 【wait_lt：等到 flag >= count 就返回】语义是「至少 count 个信号到了」。
// 适用：到达数只增不减、且「多了也没关系」的累积式流水 —— 例如归约里「第一个 peer 已经把 partials
//   初始化好」这种只要 >= 1 就能开工的场合（见 sm80_all_gather_gemm.hpp 的 kAtomic 分支）。
// 不适用：需要「恰好等于」的地方（例如复位前确认没有多计），因为它对超出不敏感。
// 【为什么只让 thread 0 自旋（L2 压力）】128 个线程同时 spin 同一行 flag，等于 128 份重复的 acquire
//   请求去抢同一条 cacheline：L2 里这条线被反复打回，对端每一次写都要分发给 128 个等者。
//   只留一个线程轮询，请求量和 L2 压力都降到 1/128。
// 【代价】其余线程被 Sync::sync() 整个挂住，CTA 在这段时间停摆、不干任何活 ——
//   所以 Flux 用 tile swizzling 去缩短「等」的时长，而不是让线程去干别的。
// 实现上的小细节：#pragma unroll 1 让编译器别把 spin 展开成一串 load —— 展开会同时压大指令 cache
//   和 L2 请求量，而这里等的是毫秒级的对端，不是几条指令的延迟。
  CUTLASS_DEVICE
  static void
  wait_lt(void *lock_ptr, int thread_idx, int flag_idx, int count) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;

    // clang-format off
    if (thread_idx == 0) {
      // Spin-loop
      #pragma unroll 1
      while (ld_acquire(flag_ptr) < count) {}
    }
    // clang-format on
    Sync::sync();
  }

// 【wait_eq：等到 flag == val】语义是「精确到齐」。
// 适用：计数从固定初值单调涨到固定终值、且写者序列唯一 —— 例如「world_size 个 rank 各 +1，等到 == world_size」。
// 风险：只要多来一次（重复 arrive、上一轮没清零、flag 被跨 iteration 复用），就永远等不到；
//   而且不会报错，只会把 kernel 挂在那里。所以凡是要复用的 flag，都该配 wait_eq_reset 或 host 侧清零。
// 与 wait_lt 只差一个比较符，但语义差一个世界：wait_lt 容忍超调，wait_eq 不容忍。
  CUTLASS_DEVICE
  static void
  wait_eq(void *lock_ptr, int thread_idx, int flag_idx, int val) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    // clang-format off
    if (thread_idx == 0) {
      #pragma unroll 1
      while (ld_acquire(flag_ptr) != val) {}
    }
    // clang-format on
    Sync::sync();
  }

// 【wait_eq_reset：等到 == val，顺手用 CAS 把它改回 reset_val】
// 一条 atomicCAS_system 同时完成「等待」和「复位」：只有当当前值确实等于 val 时写才成功，
// 于是「最后一个到齐的人」把 flag 归零，下一轮（下一次 launch / 下一个 micro-batch）从 0 重新计数。
// 为什么必须复位：这些 flag 是长期驻留的显存，不清零的话第二轮一看「已经 == world_size」就直接冲过去。
// 为什么不在 host 侧 memset：那要多一次 kernel launch、一次全 buffer 写，还得和本轮 kernel 对齐时序；
//   把复位摊给「本来就在等的那个线程」，等于免费。
// 【前提】同一个 flag 只能有一个消费者走这条路径（通常是 arrive_inc_get 里发现自己是最后一名的那位）。
//   两个消费者同时 CAS，后到的会看到 0 而永远等不到 val —— 这类死锁在小规模测试里不一定复现。
  CUTLASS_DEVICE
  static void
  wait_eq_reset(void *lock_ptr, int thread_idx, int flag_idx, int val, int reset_val = 0) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    // clang-format off
    if (thread_idx == 0) {
      #pragma unroll 1
      while(atomicCAS_system(flag_ptr, val, reset_val) != val) {}
    }
    // clang-format on
    Sync::sync();
  }

// 【arrive_inc：发一个信号】先 Sync::sync()，再让 thread 0 做 red_release。顺序本身就是语义：
//   必须等本 CTA 所有线程把数据写完（__syncthreads 保证），才允许把计数 +1；
//   否则 peer 看到计数后去读数据，读到的是「还没写完」的版本。
// 这就是 release 的实际含义 —— 不是「加个原子」，而是「我前面的写，在你看到这次加之前就已经可见」。
  CUTLASS_DEVICE
  static void
  arrive_inc(void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;

    Sync::sync();
    if (thread_idx == 0) {
      red_release(flag_ptr, val);
    }
  }

// 【arrive_inc_get：加完把新值返回】用来回答「我是不是最后一个到的」：
//   返回值 == world_size 的那个线程负责清零 / 发起下一阶段（见 sm90_reduce_scatter_utils.hpp 的用法）。
// 实现：fence + atomicAdd_system 拿旧值，+val 得到新值，再用 __shfl_sync 广播给同 warp 的其它线程。
//   「only usable for warp sync」指的就是这条广播路径：__shfl_sync 只在 warp 内有效；
//   跨 warp 想共享同一个值必须过 shared memory + __syncthreads，本函数不提供。
// 与 arrive_inc 的取舍：arrive_inc 不等返回值（快，但没人知道「到齐了」）；
//   arrive_inc_get 多等一个往返，换来「不必再找一个专门的 watcher 去轮询」。
  CUTLASS_DEVICE
  static int
  arrive_inc_get(void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
    // only usable for warp sync
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;

    Sync::sync();
    int old_val = 0;

    if (thread_idx == 0) {
      asm volatile("fence.acq_rel.sys;\n");
      old_val = atomicAdd_system(flag_ptr, val);
    }

    if constexpr (cute::is_same_v<Sync, detail::SingleThreadSync>) {
      return old_val + val;
    } else {
      int ret = __shfl_sync(0xffffffff, old_val + val, 0);
      return ret;
    }
  }
};

// 【对照组：CustomizedGenericBarrier = 同一套原语，一致性域降到 gpu】
// 差别只在两处：wait_eq_reset 用 atomicCAS（非 system），arrive_inc_get 用 fence.acq_rel.gpu + atomicAdd。
// 什么时候能这么降：flag 的写者和读者必须在同一张 GPU 上（同一 CTA、或本卡内的多个 CTA）。
// 只要有一个对端 GPU 参与，就必须用上面的 sys 版本 —— 写错不报错，只会偶发挂住。
// 把它留在同一个文件里，是为了让读者一眼看到「scope 是可选的，选错就是两套语义」。
template <class Sync>
struct CustomizedGenericBarrier : public GenericBarrier<Sync> {
  CUTLASS_DEVICE
  static void
  wait_eq_reset(void *lock_ptr, int thread_idx, int flag_idx, int val, int reset_val = 0) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    // clang-format off
    if (thread_idx == 0) {
      #pragma unroll 1
      while(atomicCAS(flag_ptr, val, reset_val) != val) {}
    }
    // clang-format on
    Sync::sync();
  }

  CUTLASS_DEVICE
  static int
  arrive_inc_get(void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
    // only usable for warp sync
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;

    Sync::sync();
    int old_val = 0;

    if (thread_idx == 0) {
      asm volatile("fence.acq_rel.gpu;\n");
      old_val = atomicAdd(flag_ptr, val);
    }

    if constexpr (cute::is_same_v<Sync, detail::SingleThreadSync>) {
      return old_val + val;
    } else {
      int ret = __shfl_sync(0xffffffff, old_val + val, 0);
      return ret;
    }
  }
};

// 【对外只暴露这一个别名】SystemBarrier = GenericSystemBarrier<SyncthreadsSync>，
// 即「整 CTA 参与 + sys scope」。warp 级/单线程场景用 GenericSystemBarrier<SingleThreadSync>，
// 两者共用同一套 PTX，只差广播方式。
using SystemBarrier = GenericSystemBarrier<detail::SyncthreadsSync>;

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace detail

}  // namespace cutlass
