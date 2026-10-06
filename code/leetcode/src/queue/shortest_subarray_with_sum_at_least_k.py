"""862. 和至少为 K 的最短子数组（Shortest Subarray with Sum at Least K）

题目：给定整数数组 nums（可能有负数）和整数 k，返回元素和 >= k 的最短非空子数组
的长度；不存在返回 -1。

思路（前缀和 + 单调队列）：
    记前缀和 P[0]=0, P[i]=nums[0..i-1] 之和。子数组 nums[i..j-1] 的和是 P[j]-P[i]，
    问题变成：找一对 i < j 使 P[j]-P[i] >= k，且 j-i 最小。

    因为数组含负数，前缀和不再单调，双指针「越右和越大」的前提不成立。
    用单调队列维护「可能成为答案左端的下标」，队列里的 P 值**严格递增**：
    - 对每个右端 j，先看队首：只要 P[j]-P[队首] >= k，就用它更新答案，
      然后把队首弹出——因为它更靠左、长度更长，对「最短」已无价值；
      注意要用 while，因为弹出若干队首后新的队首可能也满足。
    - 再把 j 从队尾压入。压入前，若队尾的 P 值 >= P[j]，说明队尾这个下标
      又靠左、前缀和又更大，将来做左端只会更差，直接弹掉，保持递增。

    每个下标最多进队、出队一次。

复杂度：时间 O(n)，空间 O(n)。
"""

from collections import deque


def shortest_subarray(nums, k):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i, x in enumerate(nums):
        prefix[i + 1] = prefix[i] + x

    ans = n + 1
    dq = deque()
    for j in range(n + 1):
        while dq and prefix[j] - prefix[dq[0]] >= k:
            ans = min(ans, j - dq.popleft())
        while dq and prefix[dq[-1]] >= prefix[j]:
            dq.pop()
        dq.append(j)
    return ans if ans <= n else -1


if __name__ == "__main__":
    assert shortest_subarray([1], 1) == 1
    assert shortest_subarray([1, 2], 4) == -1
    assert shortest_subarray([2, -1, 2], 3) == 3
    assert shortest_subarray([1, 2], 3) == 2
    assert shortest_subarray([2, 1, 2], 4) == 3
    assert shortest_subarray([84, -37, 32, 40, 95], 167) == 3
    print("shortest_subarray: all tests passed")
