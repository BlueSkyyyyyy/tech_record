"""1696. 跳跃游戏 VI（Jump Game VI）

题目：从下标 0 出发，每次最多向右跳 k 步，落在 nums[i] 就把 nums[i] 加进得分。
问到达最后一个下标能得到的最大得分。

思路（DP + 单调队列优化「窗口最大值」）：
    定义 dp[i] = 到达下标 i 时的最大得分，则
    `dp[i] = nums[i] + max(dp[j])`，其中 j 取值 i-k .. i-1，dp[0] = nums[0]。

    朴素做是 O(n*k)。注意到要的始终是「长度为 k 的滑动窗口内 dp 的最大值」，
    可以用单调队列在线维护：
    - 队首存当前窗口 dp 最大的下标；
    - 计算 dp[i] 前，先弹掉窗口外的下标（`< i-k`）；
    - 用队首 dp 算出 dp[i]，再从队尾弹掉 dp 不大于 dp[i] 的下标（它们既更靠左、
      dp 又不更大，将来不可能再成为最优），最后把 i 压入。
    答案就是 dp[n-1]。

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
    return dp[-1]


if __name__ == "__main__":
    assert max_result([1, -1, -2, 4, -7, 3], 2) == 7
    assert max_result([10, -5, -2, 4, 0, 3], 3) == 17
    assert max_result([1, -5, -20, 4, -1, 3, -6, -3], 2) == 0
    assert max_result([0], 1) == 0
    assert max_result([100, -100, -300, -300, -300, -100, 100], 4) == 0
    print("max_result: all tests passed")
