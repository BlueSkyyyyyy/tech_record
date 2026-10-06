"""1049. 最后一块石头的重量 II（Last Stone Weight II）

题目：有一堆石头，每块石头的重量都是正整数。每一回合，从中选出任意两块石头，将他们
一起粉碎。假设两块石头的重量分别为 x 和 y 且 x <= y，那么粉碎的可能结果如下：若
x == y，两块都消失；若 x != y，重 x 的那块消失，剩下重量为 y - x 的石头。最后最多
只会剩下一块石头。返回此石头可能的最小重量。

思路（0/1 背包·凑最接近一半）：
    把「粉碎」想象成给每块石头分配一个正号或负号：最终重量就是若干带符号重量之和的
    绝对值。能否让某些石头相互抵消，取决于能否把它们分成两组，使两组重量尽量相等。

    设总和为 total，我们要找一个子集，其和 s 尽量接近 total/2。这样两组之差
    (total - s) - s = total - 2s 就是最小的剩余重量。于是问题化为 0/1 背包可行性：

        dp[i] = 能否选出和为 i 的子集

    容量上界取 target = total // 2（超过一半就没意义了），每块石头倒序更新保证只用
    一次。最后从 target 向下找最大的可达和 s，返回 total - 2 * s。

    为什么是「尽量接近一半」而不是「必须相等」：总和为奇数时两边不可能完全相等，最优
    解只能取最接近的。这与 416 的区别在于——416 问「能否恰好一半」，这里问「最接近
    一半是多少」，所以 416 返回布尔，这里要扫出最大的可行容量。

复杂度：时间 O(n × total)，空间 O(total)。
"""


def last_stone_weight_ii(stones):
    total = sum(stones)
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for w in stones:
        for i in range(target, w - 1, -1):
            dp[i] = dp[i] or dp[i - w]
    for s in range(target, -1, -1):
        if dp[s]:
            return total - 2 * s
    return total


if __name__ == "__main__":
    assert last_stone_weight_ii([2, 7, 4, 1, 8, 1]) == 1
    assert last_stone_weight_ii([31, 26, 33, 21, 40]) == 5
    assert last_stone_weight_ii([1, 2]) == 1
    assert last_stone_weight_ii([1]) == 1
    assert last_stone_weight_ii([1, 1, 1]) == 1
    assert last_stone_weight_ii([2, 2]) == 0
    print("last_stone_weight_ii: all tests passed")
