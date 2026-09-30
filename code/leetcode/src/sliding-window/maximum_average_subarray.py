"""643. 子数组最大平均数 I（Maximum Average Subarray I）

题目：给定整数数组 nums 和整数 k，找出长度为 k 的连续子数组的最大平均数。
     例如 nums = [1,12,-5,-6,50,3]、k = 4，答案是 12.75。

思路：定长滑动窗口。长度为 k 的窗口，平均值最大等价于窗口和最大（k 固定），
     所以只需要维护窗口和。先算出前 k 个元素的和作为初始窗口；
     之后每次右移一格：加上新进窗口的 nums[i]，减去离开窗口的 nums[i-k]，
     用 O(1) 的代价更新窗口和，过程中记录最大值，最后除以 k。

     为什么用「滑动」而不是每个窗口重新求和：重新求和每次 O(k)，总 O(nk)；
     而窗口右移时只有一头一尾两个元素变化，其余 k-2 个被复用，所以能 O(1) 更新。

     定长窗与变长窗的区别：本题窗口长度固定为 k，右端走一步左端就跟一步；
     变长窗（3、209）则要根据条件决定左端何时收缩。这是滑动窗口的两大分支。

复杂度：时间 O(n)，空间 O(1)。
"""


def find_max_average(nums, k):
    window = sum(nums[:k])
    best = window
    for i in range(k, len(nums)):
        window += nums[i] - nums[i - k]
        best = max(best, window)
    return best / k


if __name__ == "__main__":
    assert abs(find_max_average([1, 12, -5, -6, 50, 3], 4) - 12.75) < 1e-9
    assert abs(find_max_average([5], 1) - 5.0) < 1e-9
    assert abs(find_max_average([0, 4, 0, 3, 2], 1) - 4.0) < 1e-9
    assert abs(find_max_average([-1, -2, -3, -4], 2) - (-1.5)) < 1e-9
    print("maximum_average_subarray: all tests passed")
