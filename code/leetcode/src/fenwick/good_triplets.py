"""2179. 统计数组中好三元组数目（Count Good Triplets in an Array）

题目：给定两个 0..n-1 的排列 nums1、nums2。若下标三元组 (i, j, k) 满足 i < j < k
    且在 nums1 中 nums1[i]、nums1[j]、nums1[k] 按顺序出现、在 nums2 中也按同样的
    顺序出现，则称为一个好三元组。求好三元组数目。

思路（把一个排列映射成位置数组，再数递增三元组）：
    两个排列其实是在说「同样的若干个数，在不同排列里的先后顺序」。我们想知道的是：
    能否选出三个数，它们在两个排列里的相对顺序一致。

    以 nums1 为「基准顺序」：记 pos[v] = v 在 nums1 中的下标。然后按 nums2 的顺序
    把每个数在 nums1 中的位置取出来，得到数组 b = [pos[nums2[0]], pos[nums2[1]], ...]。
    在 nums2 顺序下，一个三元组天然满足 i < j < k；要让它在 nums1 里也按顺序出现，
    等价于 b[i] < b[j] < b[k]。于是问题变成：**数 b 中的递增三元组**。

    仍然用「枚举中间人」：对每个 j，
        答案 += （b 左边比 b[j] 小的个数）×（b 右边比 b[j] 大的个数）。
    左边更小的个数用值域树状数组从左往右边扫边查；右边更大的个数 = 全局更大的个数
    − 左边更大的个数（b 是 0..n-1 的排列，所以「全局比 x 大的个数」= n - 1 - x）。

复杂度：时间 O(n log n)，空间 O(n)。
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


def good_triplets(nums1, nums2):
    n = len(nums1)
    pos = [0] * n
    for i, v in enumerate(nums1):
        pos[v] = i
    b = [pos[v] for v in nums2]

    bit = Fenwick(n)
    ans = 0
    left_count = 0
    for x in b:
        left_less = bit.prefix(x)
        left_greater = left_count - bit.prefix(x + 1)
        right_greater = (n - 1 - x) - left_greater
        ans += left_less * right_greater
        bit.add(x + 1, 1)
        left_count += 1
    return ans


if __name__ == "__main__":
    assert good_triplets([2, 0, 1, 3], [0, 1, 2, 3]) == 1
    assert good_triplets([4, 0, 1, 3, 2], [4, 1, 0, 2, 3]) == 4
    assert good_triplets([0, 1, 2], [0, 1, 2]) == 1
    assert good_triplets([0, 1, 2], [2, 1, 0]) == 0
    assert good_triplets([1, 0], [0, 1]) == 0

    print("good_triplets: all tests passed")
