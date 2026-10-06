"""2407. 最长递增子序列 II（Longest Increasing Subsequence II）

题目：给定整数数组 nums 和整数 k，求满足 i < j、nums[i] < nums[j] 且
    nums[j] - nums[i] <= k 的最长子序列长度。

思路（线段树当「值域上的最大值表」）：
    普通 LIS 的 dp[j] = 1 + max{ dp[i] : i < j, nums[i] < nums[j] }，朴素转移 O(n^2)。
    加上差值 <= k 的约束后，转移变成「在值域区间 [nums[j]-k, nums[j]-1] 里找最大的 dp」。
    于是把「值」当作下标开一棵最大值线段树：处理到 nums[j] 时，查询这个区间的最大值，
    加一得到 dp[j]，再把它写到位置 nums[j] 上（单点取 max）。答案就是全局最大值。

    为什么能「边扫边查」就自动满足 i < j：我们按数组顺序从左往右处理，
    写进树里的都是已经出现过的更早元素，天然带上了时序约束。

复杂度：O(n log V)，V 为值域上界（本题 nums[i] <= 1e5）；空间 O(V)。
"""


def length_of_lis(nums, k):
    max_v = max(nums)
    size = max_v + 1
    tree = [0] * (2 * size)

    def update(pos, val):
        i = pos + size
        if tree[i] >= val:
            return
        tree[i] = val
        i //= 2
        while i:
            tree[i] = max(tree[2 * i], tree[2 * i + 1])
            i //= 2

    def query(lo, hi):
        res = 0
        l, r = lo + size, hi + size + 1
        while l < r:
            if l & 1:
                res = max(res, tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = max(res, tree[r])
            l //= 2
            r //= 2
        return res

    ans = 0
    for v in nums:
        lo, hi = max(0, v - k), v - 1
        best = query(lo, hi) if lo <= hi else 0
        cur = best + 1
        update(v, cur)
        ans = max(ans, cur)
    return ans


if __name__ == "__main__":
    assert length_of_lis([4, 2, 1, 4, 3, 4, 5, 8, 7], 3) == 5
    assert length_of_lis([1, 5, 4, 2, 3], 2) == 3
    assert length_of_lis([7, 7, 7, 7], 1) == 1
    assert length_of_lis([1, 2, 3, 4, 5], 0) == 1
    print("longest_increasing_subsequence_ii: all tests passed")
