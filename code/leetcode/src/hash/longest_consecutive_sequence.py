"""128. 最长连续序列（Longest Consecutive Sequence）

题目：给定一个未排序的整数数组 nums，找出数字连续的最长序列的长度（不要求元素在原数组相邻）。
要求算法时间复杂度为 O(n)。例如 [100,4,200,1,3,2] 的最长连续序列是 [1,2,3,4]，长度为 4。

思路：把所有数字放进哈希集合，做到 O(1) 判断某个数在不在。
    关键技巧：只从「一段连续序列的起点」开始向后数。
    什么样的数是起点？它的前一个数 x-1 不在集合里。
    于是：
      - 若 x-1 在集合里，说明 x 不是起点，跳过（它会由更小的起点统计到）；
      - 否则从 x 开始不断 +1 检查，直到断掉，记录这一段的长度。
    为什么是 O(n)：每个数最多被「起点的向后扫描」访问一次——
    非起点被直接跳过，起点的扫描恰好覆盖各自那一段，段与段不重叠，总访问量 O(n)。
    若不做「起点判断」，每个数都往后数一遍就会退化成 O(n^2)。

复杂度：时间 O(n)，空间 O(n)。
"""


def longest_consecutive(nums):
    num_set = set(nums)
    best = 0
    for x in num_set:
        if x - 1 in num_set:
            continue
        length = 1
        while x + length in num_set:
            length += 1
        best = max(best, length)
    return best


if __name__ == "__main__":
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    assert longest_consecutive([]) == 0
    assert longest_consecutive([1, 2, 0, 1]) == 3
    print("longest_consecutive_sequence: all tests passed")
