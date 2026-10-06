"""307. 区域和检索 - 数组可修改（Range Sum Query - Mutable）

题目：实现一个数据结构，支持两种操作：
    - update(i, val)：把 nums[i] 改成 val；
    - sumRange(l, r)：返回闭区间 [l, r] 的元素和。

思路（树状数组模板）：
    静态数组的区间和用「前缀和」即可 O(1) 查询，但一旦要「单点修改」，重算后缀前缀和
    就是 O(n)。树状数组（Binary Indexed Tree, BIT）用二进制下标的巧妙分层，把
    「单点加」和「前缀和查询」都做到 O(log n)，正好补上这个缺口。

    树状数组的核心是 lowbit(i) = i & -i，即 i 的二进制最低位的 1 所代表的值。
    原数组下标从 1 开始（0 号位不用），tree[i] 维护的是区间
        (i - lowbit(i), i]
    的元素和。于是：
      - 单点加：从 i 出发不断 i += lowbit(i)，把 tree[i] 都加上 delta；
      - 前缀和：从 i 出发不断 i -= lowbit(i)，把 tree[i] 累加起来，恰好不重不漏地
        覆盖 [1, i]。

    为什么能这样：把下标用二进制拆开，i -= lowbit(i) 相当于「剥掉最低位的 1」，
    最多剥 O(log n) 次；i += lowbit(i) 相当于「把进位往上传播」，同样 O(log n) 次。

    前缀和查询区间：[l, r] 的和 = prefix(r + 1) - prefix(l)（树状数组内部用 1 为起点）。

复杂度：预处理 O(n log n)（也可 O(n) 就地建树，这里从简）；update O(log n)；
    sumRange O(log n)；空间 O(n)。
"""


class Fenwick:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s


class NumArray:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.bit = Fenwick(self.n)
        for i, x in enumerate(nums):
            self.bit.add(i + 1, x)

    def update(self, index, val):
        self.bit.add(index + 1, val - self.nums[index])
        self.nums[index] = val

    def sum_range(self, left, right):
        return self.bit.prefix(right + 1) - self.bit.prefix(left)


if __name__ == "__main__":
    a = NumArray([1, 3, 5])
    assert a.sum_range(0, 2) == 9
    a.update(1, 2)
    assert a.sum_range(0, 2) == 8
    assert a.sum_range(1, 1) == 2

    b = NumArray([0, 9, 5, 7, 3])
    assert b.sum_range(4, 4) == 3
    assert b.sum_range(2, 4) == 15
    b.update(4, 5)
    assert b.sum_range(2, 4) == 17
    b.update(0, -1)
    assert b.sum_range(0, 4) == 25

    print("range_sum_query_mutable: all tests passed")
