"""307. 区域和检索 - 数组可修改（Range Sum Query - Mutable）

题目：实现一个数据结构，支持：
    - update(i, val)：把 nums[i] 改成 val；
    - sumRange(l, r)：返回闭区间 [l, r] 的元素和。

思路（线段树模板 · 单点修改 + 区间查询）：
    静态数组的区间和用「前缀和」就能 O(1) 查询，但单点修改后要重算后缀，退化到 O(n)。
    线段树把数组分成一棵完全二叉树：叶子是单个元素，内部节点维护「两个孩子的合并值」。
    对本  题合并值就是「和」。单点修改从叶子往上更新 O(log n) 个节点；
    区间查询把 [l, r] 拆成 O(log n) 个整块再合并。这是一切线段树的母版。

    这里用「自底向上」的迭代写法（数组存储 size = n，叶子在 [n, 2n)）：
      - 单点改：改叶子，然后 i //= 2 一路向上重算；
      - 区间查：把左闭右开 [l, r) 映射成叶子下标，用两个游标向中间/向上夹，
        左游标是奇数就说明它不能被父节点代表，必须单独取走，右游标同理。

复杂度：建树 O(n)；update O(log n)；sumRange O(log n)；空间 O(n)。
"""


class NumArray:
    def __init__(self, nums):
        self.n = len(nums)
        self.tree = [0] * (2 * self.n)
        for i, v in enumerate(nums):
            self.tree[self.n + i] = v
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, index, val):
        i = index + self.n
        self.tree[i] = val
        i //= 2
        while i:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def sumRange(self, left, right):
        res = 0
        l, r = left + self.n, right + self.n + 1
        while l < r:
            if l & 1:
                res += self.tree[l]
                l += 1
            if r & 1:
                r -= 1
                res += self.tree[r]
            l //= 2
            r //= 2
        return res


if __name__ == "__main__":
    a = NumArray([1, 3, 5])
    assert a.sumRange(0, 2) == 9
    a.update(1, 2)
    assert a.sumRange(0, 2) == 8
    assert a.sumRange(1, 1) == 2

    b = NumArray([0, 9, 5, 7, 3])
    assert b.sumRange(4, 4) == 3
    assert b.sumRange(2, 4) == 15
    b.update(4, 5)
    assert b.sumRange(2, 4) == 17
    b.update(0, -1)
    assert b.sumRange(0, 4) == 25

    c = NumArray([-1])
    assert c.sumRange(0, 0) == -1
    c.update(0, 7)
    assert c.sumRange(0, 0) == 7

    print("range_sum_query_mutable: all tests passed")
