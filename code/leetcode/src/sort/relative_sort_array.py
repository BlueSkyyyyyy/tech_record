"""1122. 数组的相对排序（Relative Sort Array）

题目：给你两个数组 arr1 和 arr2。arr2 中的元素互不相同，且 arr2 中每个元素都
      出现在 arr1 中。请把 arr1 排序，使得 arr1 中元素的相对顺序与 arr2 中的
      相对顺序一致；未在 arr2 中出现的元素，按升序排在末尾。

思路（把「顺序」翻译成一个可比较的键）：
    题目要的不是普通的大小排序，而是「先按 arr2 指定的顺序，再对剩下的升序」。
    这类「自定义顺序」的通用技巧，是给每个值造一个排序键：某个值在 arr2 里的下标
    就是它的优先级，越小越靠前；不在 arr2 里的值统一给一个大优先级（比如
    len(arr2)），它们之间再用数值本身升序打破平局。

    键写成一个二元组 `(优先级, 数值)`，直接交给语言自带的排序即可：先比优先级，
    优先级相同再比数值。这样一行就把两种规则合并了。

    另一种更快的写法是计数排序：先用桶统计 arr1 各值出现次数，按 arr2 顺序输出对应
    次数，再把剩下的非零值按升序输出，时间 O(n + m + range)。当数值范围很大时，
    上面的比较排序 O(n log n) 更通用；数值范围小则计数更快。
"""


def relative_sort_array(arr1, arr2):
    rank = {v: i for i, v in enumerate(arr2)}
    return sorted(arr1, key=lambda x: (rank.get(x, len(arr2)), x))


def relative_sort_array_counting(arr1, arr2):
    from collections import Counter

    cnt = Counter(arr1)
    res = []
    for v in arr2:
        res.extend([v] * cnt.pop(v, 0))
    for v in sorted(cnt):
        res.extend([v] * cnt[v])
    return res


if __name__ == "__main__":
    a1 = [2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19]
    a2 = [2, 1, 4, 3, 9, 6]
    want = [2, 2, 2, 1, 4, 3, 3, 9, 6, 7, 19]
    assert relative_sort_array(a1, a2) == want
    assert relative_sort_array_counting(a1, a2) == want

    assert relative_sort_array([28, 6, 22, 8, 44, 17], [22, 28, 8, 6]) == [
        22, 28, 8, 6, 17, 44,
    ]
    assert relative_sort_array([1, 2, 3], []) == [1, 2, 3]
    print("relative_sort_array: all tests passed")
