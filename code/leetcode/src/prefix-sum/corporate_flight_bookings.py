"""1109. 航班预订统计（Corporate Flight Bookings）

题目：有 n 个航班，编号 1..n。给定预订记录 bookings，每条是 [first, last, seats]，
      表示从 first 到 last（含两端）每个航班都被预订了 seats 个座位。
      返回长度为 n 的数组 answer，answer[i] 是第 i+1 个航班的总预订数。
      例如 n = 5，bookings = [[1,2,10],[2,3,20],[2,5,25]]，
      结果为 [10, 55, 45, 25, 25]。

思路：如果每条预订都去把它覆盖的每个航班逐个加上 seats，最坏 O(n * m)。
      这类「多次给一段区间整体加同一个数，最后问每个位置的值」正是差分的用武之地。
      记差分数组 diff，对区间 [l, r] 加 v 只做两件事：
          diff[l] += v        # 从 l 开始，增量在之后持续生效
          diff[r + 1] -= v    # 到 r 之后，增量被抵消，不再生效
      这样一次区间修改只要 O(1)。等所有修改做完，对 diff 求一次前缀和，
      就还原出每个位置实际累计到的值。换句话说：差分把「区间加」变成「两个单点改」，
      而「单点改求前缀和」正是前缀和的逆过程。
      本题输入是 1 下标，内部转成 0 下标：diff[first-1] += seats，diff[last] -= seats
      （last 对应 0 下标的 last-1，其下一位即 last，故写 diff[last]）。

复杂度：时间 O(n + m)（m 为预订条数），空间 O(n)。
"""


def corp_flight_bookings(bookings, n):
    diff = [0] * (n + 1)
    for first, last, seats in bookings:
        diff[first - 1] += seats
        diff[last] -= seats
    res = [0] * n
    cur = 0
    for i in range(n):
        cur += diff[i]
        res[i] = cur
    return res


if __name__ == "__main__":
    got = corp_flight_bookings([[1, 2, 10], [2, 3, 20], [2, 5, 25]], 5)
    want = [10, 55, 45, 25, 25]
    assert got == want, got
    assert corp_flight_bookings([], 3) == [0, 0, 0]
    assert corp_flight_bookings([[1, 1, 7]], 1) == [7]
    assert corp_flight_bookings([[2, 2, 5], [2, 2, 3]], 3) == [0, 8, 0]
    print("corporate_flight_bookings: all tests passed")
