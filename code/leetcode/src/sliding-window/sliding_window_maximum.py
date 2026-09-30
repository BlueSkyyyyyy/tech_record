"""239. 滑动窗口最大值（Sliding Window Maximum）

题目：给定整数数组 nums 和窗口大小 k，窗口从最左滑到最右，返回每个窗口内的最大值组成的数组。
      例如 nums = [1,3,-1,-3,5,3,6,7]、k = 3，答案是 [3,3,5,5,6,7]。

思路：单调队列（双端队列 deque）。队列里存**下标**，并始终保持下标对应的值从上到下递减。
      右端进来一个 x 时，先把队尾所有 <= x 的值弹掉——它们既不可能是当前窗口最大值，
      也不可能比 x 更晚失效，留着没有意义；然后把当前下标入队。
      再看队首：如果队首下标已经滑出窗口（<= i - k），就从前面弹出。此时队首就是当前窗口最大值的下标。

      为什么队列要存下标而不是值：需要判断队首是否已滑出窗口，只有下标能给出这个信息。

      为什么每个元素只进出队一次、整体 O(n)：每个下标最多入队一次；被弹掉的下标不会
      再回来。虽然内层有 while，但它执行的总次数不超过入队总次数，所以摊还 O(n)。

      为什么「弹掉队尾更小的值」是对的：窗口在向右滑动，越靠右的元素活得越久。新来的 x
      比队尾的旧值大，那么在 x 失效之前，旧值永远不可能是最大值；等 x 失效时，旧值早已
      滑出窗口。所以旧值可以直接丢弃。

复杂度：时间 O(n)，空间 O(k)。
"""
from collections import deque


def max_sliding_window(nums, k):
    dq = deque()
    res = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res


if __name__ == "__main__":
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window([1], 1) == [1]
    assert max_sliding_window([1, -1], 1) == [1, -1]
    assert max_sliding_window([9, 11], 2) == [11]
    assert max_sliding_window([4, 2, 12, 3, 5], 3) == [12, 12, 12]
    print("sliding_window_maximum: all tests passed")
