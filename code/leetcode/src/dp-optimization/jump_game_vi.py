"""1696. 跳跃游戏 VI（Jump Game VI）

题目：从下标 0 出发，每次最多向右跳 k 步，最后必须到达 n-1。
得分是经过位置的 nums 之和（含起点终点），求最大得分。

思路（DP + 单调队列优化）：
    设 dp[i] = 到达下标 i 的最大得分：
        dp[i] = nums[i] + max{ dp[j] : i-k <= j < i }，dp[0] = nums[0]。
    「前 k 个 dp 的最大值」用单调递减双端队列维护，队首即最大值，整体 O(n)。

复杂度：时间 O(n)，空间 O(n)。
"""

from collections import deque


def max_result(nums, k):
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dq = deque([0])
    for i in range(1, n):
        while dq and dq[0] < i - k:
            dq.popleft()
        dp[i] = nums[i] + dp[dq[0]]
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
    return dp[n - 1]


if __name__ == "__main__":
    assert max_result([1, -1, -2, 4, -7, 3], 2) == 7
    assert max_result([10, -5, -2, 4, 0, 3], 3) == 17
    assert max_result([1, -5, -20, 4, -1, 3, -6, -3], 2) == 0
    assert max_result([5], 1) == 5
    print("jump_game_vi: all tests passed")
