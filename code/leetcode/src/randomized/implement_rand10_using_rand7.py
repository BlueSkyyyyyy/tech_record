"""470. 用 Rand7() 实现 Rand10()（Implement Rand10() Using Rand7()）

题目：已有 rand7()，它等概率返回 1..7 的整数。只允许调用 rand7()，实现 rand10()，
    等概率返回 1..10 的整数。

思路（拒绝采样）：
    一次 rand7 只有 7 个结果，凑不出 10 的等概率。把两次调用组合起来：
        r = (rand7() - 1) * 7 + rand7()
    它把结果均匀地铺在 1..49 上（相当于一个七进制两位数）。只要 r ≤ 40，就返回
    (r - 1) % 10 + 1——40 是 10 的倍数，所以 1..40 均分到 1..10，每类恰好 4 个，
    等概率。若 r 落在 41..49（9 个数），就拒绝并重新抽。

    为什么要「拒绝」而不是把 41..49 折回去？折回会破坏均匀性。拒绝采样虽然有时要重抽，
    但保证结果严格均匀。期望调用次数 = 2 × 49/40 ≈ 2.45 次。

复杂度：期望时间 O(1)，空间 O(1)。
"""

import random


def rand7():
    return random.randint(1, 7)


def rand10():
    while True:
        r = (rand7() - 1) * 7 + rand7()
        if r <= 40:
            return (r - 1) % 10 + 1


if __name__ == "__main__":
    random.seed(0)
    counts = [0] * 11
    for _ in range(100000):
        x = rand10()
        assert 1 <= x <= 10
        counts[x] += 1
    for x in range(1, 11):
        assert abs(counts[x] / 100000 - 0.1) < 0.02
    print("implement_rand10_using_rand7: all tests passed")
