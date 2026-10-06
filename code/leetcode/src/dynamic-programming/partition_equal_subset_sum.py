"""416. 分割等和子集（Partition Equal Subset Sum）

题目：给你一个只包含正整数的非空数组 nums，判断能否把它分割成两个子集，使得两个
子集的元素和相等。

思路（0/1 背包·可行性）：
    「两个子集和相等」等价于「存在一个子集的和恰好是整个数组和的一半」。设总和为
    total，若 total 是奇数则直接返回 False；否则问题变成：能否从 nums 中挑出若干
    个数，使它们的和恰好为 target = total // 2。

    设 dp[i] 表示「能否从已处理的数中选出和为 i 的子集」。每遇到一个数 num，它要么
    被选、要么不被选：

        dp[i] = dp[i] or dp[i - num]   （i 从 target 递减到 num）

    初始化 dp[0] = True（和为 0 的子集就是空集）。答案 dp[target]。

    为什么容量要**倒序**：这是 0/1 背包与完全背包的分水岭。正序时 dp[i-num] 可能已经
    被本轮更新过，等于同一个 num 被用了两次；倒序保证用到的是「本轮之前」的旧值，
    从而每个数至多用一次。经典 0/1 背包「一维数组倒序」的由来就在这里。

复杂度：时间 O(n × target)，空间 O(target)。
"""


def can_partition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for i in range(target, num - 1, -1):
            dp[i] = dp[i] or dp[i - num]
    return dp[target]


if __name__ == "__main__":
    assert can_partition([1, 5, 11, 5]) is True
    assert can_partition([1, 2, 3, 5]) is False
    assert can_partition([1, 1]) is True
    assert can_partition([2, 2, 2]) is False
    assert can_partition([1, 2, 5]) is False
    print("partition_equal_subset_sum: all tests passed")
