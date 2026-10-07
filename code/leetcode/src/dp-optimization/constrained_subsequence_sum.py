"""1425. 带限制的子序列和（Constrained Subsequence Sum）

题目：给定数组 nums 和整数 k，选一个非空子序列，使「相邻被选元素在原数组中的下标差 <= k」，
且子序列和最大。返回这个最大和。

思路（DP + 单调队列优化）：
    设 dp[i] = 以 nums[i] 结尾的最大子序列和：
        dp[i] = nums[i] + max(0, max{ dp[j] : i-k <= j < i })
    max(0, …) 表示可以从 nums[i] 重新开始，不接任何前驱。
    瓶颈是窗口最大值：窗口大小固定为 k。用一个**单调递减双端队列**维护下标，
    队首就是窗口内 dp 的最大值，整体 O(n)：
      - 计算前：把队首下标 < i-k 的弹出；
      - 取 dp[队首]（或 0）转移；
      - 入队前把队尾 dp 值 <= 当前 dp 的弹出，保证单调递减。

复杂度：时间 O(n)，空间 O(n)。
"""

from collections import deque


def constrained_subset_sum(nums, k):
    n = len(nums)
    dp = [0] * n
    dq = deque()
    best = nums[0]
    for i in range(n):
        while dq and dq[0] < i - k:
            dq.popleft()
        prev = max(0, dp[dq[0]]) if dq else 0
        dp[i] = nums[i] + prev
        best = max(best, dp[i])
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
    return best


if __name__ == "__main__":
    assert constrained_subset_sum([10, 2, -10, 5, 20], 2) == 37
    assert constrained_subset_sum([-1, -2, -3], 1) == -1
    assert constrained_subset_sum([10, -2, -10, -5, 20], 2) == 23
    assert constrained_subset_sum([-3], 1) == -3
    print("constrained_subsequence_sum: all tests passed")
