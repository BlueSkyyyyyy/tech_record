"""875. 爱吃香蕉的珂珂（Koko Eating Bananas）

题目：有 n 堆香蕉，第 i 堆有 piles[i] 根。珂珂每小时选择一堆，吃 k 根；如果这堆不足 k 根，
      她吃完这堆后这一小时不再吃别的（即每小时最多吃一堆）。给定警卫将在 h 小时后回来，
      求她能在 h 小时内吃完所有香蕉的最小速度 k（k 为整数，根/小时）。h >= piles.length。

思路：这是典型的**二分答案**。我们不枚举「怎么吃」，而是直接猜「速度 k」，然后验证在速度 k 下
      能否按时吃完。关键观察是**单调性**：速度越大，耗时越少。所以「能按时吃完」这个判定条件
      随着 k 增大从假变真，存在一个临界点，正是答案。

      验证函数 can_finish(k)：第 i 堆需要 ceil(piles[i] / k) 小时，
      整数上取整写成 (piles[i] + k - 1) // k；总时间不超过 h 即成功。

      答案的取值区间：下界是 1（不能不吃），上界是 max(piles)（再快也只能一小时吃一堆，
      比最大堆还快没有意义）。在这个区间上二分第一个「可行」的 k：可行就 right = mid（保留 mid，
      因为可能有更小的可行值），不可行就 left = mid + 1。

复杂度：时间 O(n log(max(piles)))，空间 O(1)。
"""


def min_eating_speed(piles, h):
    def can_finish(k):
        hours = 0
        for pile in piles:
            hours += (pile + k - 1) // k
        return hours <= h

    left, right = 1, max(piles)
    while left < right:
        mid = left + (right - left) // 2
        if can_finish(mid):
            right = mid
        else:
            left = mid + 1
    return left


if __name__ == "__main__":
    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert min_eating_speed([30, 11, 23, 4, 20], 5) == 30
    assert min_eating_speed([30, 11, 23, 4, 20], 6) == 23
    assert min_eating_speed([1], 1) == 1
    assert min_eating_speed([312884470], 312884469) == 2
    assert min_eating_speed([1000000000], 2) == 500000000
    print("koko_eating_bananas: all tests passed")
