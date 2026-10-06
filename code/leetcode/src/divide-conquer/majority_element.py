"""169. 多数元素（Majority Element）

题目：给定一个大小为 n 的数组 nums，返回其中的多数元素。多数元素是指在数组中
出现次数大于 n/2 的元素。你可以假设数组非空，且给定的数组总是存在多数元素。

思路（分治：拆成两半，各自找候选，再合并）：
    分治三步走：
      1. 分解：把数组从中间切成左右两半；
      2. 解决：递归地分别求出左半段和右半段的多数候选；
      3. 合并：如果左右候选相同，它显然就是整段的候选；如果不同，就数一数
         两者在整段里各出现多少次，出现更多的那个留下。

    为什么合并规则成立：整段的多数元素出现次数 > n/2，它必然在左半段或右半段里
    也是多数（否则两半都不超过一半，加起来也到不了 n/2）。所以整段的多数一定
    出现在「左半候选」或「右半候选」之中，我们只需在这两个候选里比票数，
    不必检查其它元素。

    为什么不必用哈希计数一次性统计：分治版本不依赖额外空间，结构清晰，是
    「分解 → 解决 → 合并」最干净的入门例子。它的代价是 O(n log n)，比
    投票法（Boyer-Moore）的 O(n) 慢，但更适合用来体会分治的形状。

复杂度：时间 O(n log n)（递归树每层合计扫描 O(n)，共 log n 层），
    空间 O(log n)（递归栈深度）。
"""


def majority_element(nums):
    def count_in_range(lo, hi, target):
        count = 0
        for i in range(lo, hi):
            if nums[i] == target:
                count += 1
        return count

    def majority(lo, hi):
        if hi - lo == 1:
            return nums[lo]
        mid = (lo + hi) // 2
        left = majority(lo, mid)
        right = majority(mid, hi)
        if left == right:
            return left
        left_count = count_in_range(lo, hi, left)
        right_count = count_in_range(lo, hi, right)
        return left if left_count > right_count else right

    return majority(0, len(nums))


if __name__ == "__main__":
    assert majority_element([3, 2, 3]) == 3
    assert majority_element([2, 2, 1, 1, 1, 2, 2]) == 2
    assert majority_element([1]) == 1
    assert majority_element([1, 1, 2]) == 1
    assert majority_element([1, 2, 1]) == 1
    print("majority_element: all tests passed")
