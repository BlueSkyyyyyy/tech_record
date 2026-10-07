"""1235. 规划兼职工作（Maximum Profit in Job Scheduling）

题目：给定 startTime、endTime、profit，选出不重叠的若干工作使总报酬最大。
工作 i 在 [startTime[i], endTime[i]] 进行，结束时刻与另一工作开始时刻相同不算重叠。

思路（DP + 二分查找优化）：
    先按结束时间排序，设 dp[i] = 只考虑前 i 个工作（1-indexed）能拿到的最大报酬。
    对第 i 个工作 (s, e, p)，有两种选择：
      - 不做：dp[i-1]；
      - 做：需要找「结束时间 <= s」的最后一个工作 m，则 dp[m] + p。
    因为工作已按结束时间排序，用 `bisect_right(ends, s, 0, i-1)` 在 O(log n) 内
    定位 m，避免 O(n) 线性回溯。
        dp[i] = max(dp[i-1], dp[m] + p)

复杂度：时间 O(n log n)（排序 + 每个工作一次二分），空间 O(n)。
"""

import bisect


def job_scheduling(start_time, end_time, profit):
    jobs = sorted(zip(end_time, start_time, profit))
    ends = [e for e, _, _ in jobs]
    n = len(jobs)
    dp = [0] * (n + 1)
    for i, (_e, s, p) in enumerate(jobs, 1):
        m = bisect.bisect_right(ends, s, 0, i - 1)
        dp[i] = max(dp[i - 1], dp[m] + p)
    return dp[n]


if __name__ == "__main__":
    assert job_scheduling([1, 2, 3, 3], [3, 4, 5, 6], [50, 10, 40, 70]) == 120
    assert job_scheduling([1, 2, 3, 4, 6], [3, 5, 10, 6, 9], [20, 20, 100, 70, 60]) == 150
    assert job_scheduling([1, 1, 1], [2, 3, 4], [5, 6, 7]) == 7
    print("job_scheduling: all tests passed")
