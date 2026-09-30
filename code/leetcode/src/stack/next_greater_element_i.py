"""496. 下一个更大元素 I（Next Greater Element I）

题目：nums1 是 nums2 的子集。对 nums1 中的每个元素 x，找出它在 nums2 中
      对应位置右侧第一个比它大的元素；不存在则输出 -1。

思路（先在 nums2 上跑单调栈，把「每个值 → 它的下一个更大值」存进哈希表）：
    遍历 nums2，维护一个单调递减栈。遇到比栈顶大的 x，就不断弹出栈顶 v，
    记下 next_greater[v] = x；最后把 x 压栈。遍历完 nums2 后，
    哈希表里就装好了 nums2 中所有「有下一个更大值」的元素的答案。
    再按 nums1 的顺序查表，查不到就是 -1。

    为什么可以先只处理 nums2：nums1 只是 nums2 的一个查询子集。
    与其对 nums1 每个元素都去 nums2 里找位置再向后扫（慢），
    不如一次把 nums2 的答案全算好，之后每次查询都是 O(1)。
    这就是「预处理 + 哈希查询」的空间换时间。

    这里栈里存的是「值」而不是下标：因为输出的是值本身，且 nums2 无重复元素，
    不需要靠下标去重，直接存值更直观。

复杂度：时间 O(n + m)（n、m 分别是 nums2、nums1 长度），空间 O(n)（哈希表 + 栈）。
"""


def next_greater_element(nums1, nums2):
    next_greater = {}
    stack = []
    for x in nums2:
        while stack and stack[-1] < x:
            next_greater[stack.pop()] = x
        stack.append(x)
    return [next_greater.get(x, -1) for x in nums1]


if __name__ == "__main__":
    assert next_greater_element([4, 1, 2], [1, 3, 4, 2]) == [-1, 3, -1]
    assert next_greater_element([2, 4], [1, 2, 3, 4]) == [3, -1]
    assert next_greater_element([1, 3, 5], [6, 5, 4, 3, 2, 1, 7]) == [7, 7, 7]
    print("next_greater_element: all tests passed")
