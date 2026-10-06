"""1438. 绝对差不超过限制的最长连续子数组

题目：给整数数组 nums 和 limit，返回最长的连续子数组，使其中任意两元素之差的
绝对值都不超过 limit。

思路（滑动窗口 + 两条单调队列）：
    「窗口内任意两元素之差 <= limit」等价于「窗口最大值 - 窗口最小值 <= limit」。
    于是右端一路向右扩，用两条单调队列分别实时维护窗口的最大值和最小值：
    - maxq：从队首到队尾单调**递减**，队首即当前窗口最大值；
    - minq：从队首到队尾单调**递增**，队首即当前窗口最小值。
    每次把新元素压入前，从队尾弹掉不可能再成为极值的元素（比新元素小/大的旧值）。
    若此时「最大值 - 最小值 > limit」，就移动左端收缩：当窗口左端元素正好等于某条
    队列的队首时，把它从队首弹出。
    收缩到合法后，用 `right - left + 1` 更新答案。

    为什么左端只在「等于队首」时弹：队首就是当前极值本身，左端元素若不是极值，
    它早就被后续更大/更小的元素从队尾挤掉了，不出现在队首上。

复杂度：时间 O(n)（每个元素进出各队列至多一次），空间 O(n)。
"""

from collections import deque


def longest_subarray(nums, limit):
    maxq = deque()
    minq = deque()
    left = 0
    ans = 0
    for right, x in enumerate(nums):
        while maxq and maxq[-1] < x:
            maxq.pop()
        maxq.append(x)
        while minq and minq[-1] > x:
            minq.pop()
        minq.append(x)

        while maxq[0] - minq[0] > limit:
            if maxq[0] == nums[left]:
                maxq.popleft()
            if minq[0] == nums[left]:
                minq.popleft()
            left += 1

        ans = max(ans, right - left + 1)
    return ans


if __name__ == "__main__":
    assert longest_subarray([8, 2, 4, 7], 4) == 2
    assert longest_subarray([10, 1, 2, 4, 7, 2], 5) == 4
    assert longest_subarray([4, 2, 2, 2, 4, 4, 2, 2], 0) == 3
    assert longest_subarray([1], 0) == 1
    assert longest_subarray([1, 5], 3) == 1
    print("longest_subarray: all tests passed")
