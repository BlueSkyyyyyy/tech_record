"""1636. 按照频率将数组升序排序（Sort Array by Increasing Frequency）

题目：给你一个整数数组 nums，请按每个值出现的频率升序排序数组；如果两个值的频率
      相同，则按它们的数值降序排列。返回排序后的数组。

思路（先统计，再用「频率 + 数值」当排序键）：
    排序依据不是元素本身，而是「这个元素出现了几次」。所以第一步必须先把频率数出来，
    用哈希/计数器 `cnt` 记录每个值出现的次数。

    接着对整个数组排序，键写成 `(cnt[x], -x)`：先比频率，频率小的在前（升序）；
    频率相同时比 `-x`，即数值大的在前（因为对 `-x` 升序等于对 `x` 降序）。

    为什么直接排原数组而不是排「(频率, 值)」的集合：同一个值要重复出现它应有的次数，
    原数组恰好是「每个值展开成重复元素」的现成载体，按上述键排序即可。

    这是「多级排序键」的标准用法：把若干条规则依次写进元组，元组比较天然按字典序
    逐级进行，想升序就写原值，想降序就写相反数（或在支持的写法里用 reverse 分段）。
"""


def frequency_sort(nums):
    from collections import Counter

    cnt = Counter(nums)
    return sorted(nums, key=lambda x: (cnt[x], -x))


if __name__ == "__main__":
    assert frequency_sort([1, 1, 2, 2, 2, 3]) == [3, 1, 1, 2, 2, 2]
    assert frequency_sort([2, 3, 1, 3, 2]) == [1, 3, 3, 2, 2]
    assert frequency_sort([-1, 1, -6, 4, 5, -6, 1, 4, 1]) == [
        5, -1, 4, 4, -6, -6, 1, 1, 1,
    ]
    assert frequency_sort([7]) == [7]
    print("frequency_sort: all tests passed")
