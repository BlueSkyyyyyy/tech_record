"""718. 最长重复子数组（Maximum Length of Repeated Subarray）

题目：给两个整数数组 nums1 和 nums2，返回两个数组中公共的、长度最长的连续子数组
的长度。

思路（二维 DP：以「这两个位置」结尾的公共后缀）：
    和 1143 一样是两个数组，但这次要求「连续」。连续性同样提示把状态钉在结尾：

        dp[i][j] = nums1 以第 i 个元素结尾、nums2 以第 j 个元素结尾的最长公共后缀长度。

    只有当两个结尾元素相等时，这一段公共后缀才有意义，且可以在去掉这两个结尾的
    基础上再接一格：

        nums1[i-1] == nums2[j-1]  ->  dp[i][j] = dp[i-1][j-1] + 1
        否则                      ->  dp[i][j] = 0

    和 1143 的关键区别：这里不相等时**不能**从 dp[i-1][j] / dp[i][j-1] 继承，因为
    公共子数组必须连续，一旦结尾对不上，以这对位置结尾的公共后缀长度只能归零。
    答案要取整张表的最大值，而不是 dp[m][n]（最长的那段可能停在任意位置）。

复杂度：时间 O(m·n)，空间 O(m·n)（可滚动到 O(n)）。
"""


def find_length(nums1, nums2):
    m, n = len(nums1), len(nums2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if nums1[i - 1] == nums2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best


if __name__ == "__main__":
    assert find_length([1, 2, 3, 2, 1], [3, 2, 1, 4, 7]) == 3
    assert find_length([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]) == 5
    assert find_length([1, 2, 3], [4, 5, 6]) == 0
    assert find_length([], [1, 2]) == 0
    assert find_length([5], [5]) == 1
    print("maximum_length_of_repeated_subarray: all tests passed")
