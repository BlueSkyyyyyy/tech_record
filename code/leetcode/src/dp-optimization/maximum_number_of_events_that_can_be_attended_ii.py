"""1751. 最多可以参加的会议数目 II（Maximum Number of Events That Can Be Attended II）

题目：events[i] = [start, end, value]，同一时间只能参加一个会议（区间不能重叠，
后一个的 start 必须严格大于前一个的 end）。最多参加 k 个会议，求能获得的最大价值。

思路（按结束时间排序 + 二分找「可接的前一个」 + DP）：
    这是「加权区间调度」的 k 层版本。先按结束时间升序排序，问题立刻变成「从左往右
    选，选了第 i 个之后只能接在它后面」：
        dp[i][j] = 考虑前 i 个会议、最多选 j 个时的最大价值
    对第 i 个会议，只有两个选择：
        不选：dp[i-1][j]
        选：  dp[p][j-1] + value_i，其中 p 是「结束时间 < start_i」的最后一个会议下标
    求 p 用二分（结束时间已排序），把朴素 O(n) 的「枚举前驱」降成 O(log n)。
    不选时保留 dp[i-1][j]，所以「最多 k 个」自然包含「少于 k 个」的情形。

复杂度：排序 O(n log n)，DP O(n · k · log n)，空间 O(n · k)。
"""

from bisect import bisect_left


def max_value(events, k):
    events = sorted(events, key=lambda e: e[1])
    n = len(events)
    ends = [e[1] for e in events]

    # dp[i][j]：前 i 个会议、最多选 j 个
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        start, _end, value = events[i - 1]
        p = bisect_left(ends, start)  # 结束时间 < start 的会议个数
        for j in range(1, k + 1):
            dp[i][j] = max(dp[i - 1][j], dp[p][j - 1] + value)
    return dp[n][k]


if __name__ == "__main__":
    assert max_value([[1, 2, 4], [3, 4, 3], [2, 3, 1]], 2) == 7
    assert max_value([[1, 2, 4], [3, 4, 3], [2, 3, 10]], 2) == 10
    assert max_value([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], 3) == 9
    assert max_value([[1, 3, 5]], 1) == 5
    print("maximum_number_of_events_that_can_be_attended_ii: all tests passed")
