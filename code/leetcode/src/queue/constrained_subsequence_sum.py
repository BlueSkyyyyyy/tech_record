"""1425. 带限制的子序列和（Constrained Subsequence Sum）

题目：给整数数组 nums 和整数 k，求一个非空子序列的最大和，要求子序列中相邻两个
元素在原数组里的下标之差不超过 k。

思路（DP + 单调队列）：
    定义 dp[i] = 以 nums[i] 结尾的合法子序列的最大和。它可以自己单独成段，也可以接在
    前面某个满足 `i-k <= j < i` 的 dp[j] 后面：
    `dp[i] = nums[i] + max(0, max(dp[j]))`，j 在 i-k .. i-1。
    取 `max(0, ...)` 是因为「接在前面」是可选的：前面那段和为负时，不如从自己重新开始。
    答案取所有 dp[i] 的最大值（不一定是最后一个）。

    「窗口内 dp 最大值」用单调队列维护：计算 dp[i] 前弹掉下标 `< i-k` 的队首，
    队首即为窗口内 dp 最大者；算完 dp[i] 后从队尾弹掉 dp 不大于它的下标再入队。

复杂度：时间 O(n)，空间 O(n)。
"""

from collections import deque


def constrained_subset_sum(nums, k):
    n = len(nums)
    dp = [0] * n
    dq = deque()
    ans = nums[0]
    for i in range(n):
        while dq and dq[0] < i - k:
            dq.popleft()
        best = max(0, dp[dq[0]]) if dq else 0
        dp[i] = nums[i] + best
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
        ans = max(ans, dp[i])
    return ans


if __name__ == "__main__":
    assert constrained_subset_sum([10, 2, -10, 5, 20], 2) == 37
    assert constrained_subset_sum([-1, -2, -3], 1) == -1
    assert constrained_subset_sum([10, -2, -10, -5, 20], 2) == 23
    assert constrained_subset_sum([-5], 1) == -5
    assert constrained_subset_sum([1, 2, 3, 4, 5], 2) == 15
    print("constrained_subset_sum: all tests passed")
