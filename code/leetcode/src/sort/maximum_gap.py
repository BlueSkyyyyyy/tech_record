"""164. 最大间距（Maximum Gap）

题目：给定一个无序整数数组 nums，返回排序后相邻元素之间差值的最大值。若数组元素
      少于 2 个，返回 0。要求在线性时间内完成。

思路（桶排序 + 鸽巢原理）：
    直接排序再求相邻差是 O(n log n)，题目要求线性，必须换思路。

    设数组最小值 lo、最大值 hi，元素个数 n。把区间 [lo, hi] 等分成若干个桶，桶内
    只记录该桶出现过的**最小值和最大值**（不记录全部元素）。桶宽取
    `size = max(1, (hi - lo) // (n - 1))`，桶数 `(hi - lo) // size + 1`。

    关键结论：**最大间距一定出现在相邻两个「非空桶」的「后一桶最小值 − 前一桶最大值」
    之间**，而不会在同一桶内。为什么？鸽巢原理：n 个数被放进若干桶，若最大间距发生在
    同一桶内，则该桶内两个元素的差小于桶宽 size；但按 size 的定义，n-1 个间距的平均值
    是 (hi-lo)/(n-1)，而桶宽不超过这个平均值，于是至少有一个间距不小于桶宽——矛盾。
    更直接的看法是：桶宽 ≤ 平均间距，所以真正的最大间距必然跨桶。

    于是只扫一遍桶，维护「上一个非空桶的最大值 prev」，用当前桶的最小值减 prev 更新
    答案，再让 prev 变为当前桶的最大值。初始化 prev = lo，保证第一个非空桶也能正确
    与下界比较。

    边界：元素少于 2 个返回 0；所有元素相等（lo == hi）也返回 0，否则除法会出问题。
"""


def maximum_gap(nums):
    if len(nums) < 2:
        return 0
    n = len(nums)
    lo, hi = min(nums), max(nums)
    if lo == hi:
        return 0
    size = max(1, (hi - lo) // (n - 1))
    count = (hi - lo) // size + 1
    buckets = [[None, None] for _ in range(count)]
    for x in nums:
        idx = (x - lo) // size
        b = buckets[idx]
        b[0] = x if b[0] is None else min(b[0], x)
        b[1] = x if b[1] is None else max(b[1], x)
    best = 0
    prev = lo
    for bmin, bmax in buckets:
        if bmin is None:
            continue
        best = max(best, bmin - prev)
        prev = bmax
    return best


if __name__ == "__main__":
    assert maximum_gap([3, 6, 9, 1]) == 3
    assert maximum_gap([10]) == 0
    assert maximum_gap([]) == 0
    assert maximum_gap([1, 10000000]) == 9999999
    assert maximum_gap([1, 1, 1, 1]) == 0
    assert maximum_gap([1, 2, 3, 4, 5]) == 1
    print("maximum_gap: all tests passed")
