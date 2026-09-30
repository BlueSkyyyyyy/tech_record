"""35. 搜索插入位置（Search Insert Position）

题目：给定一个升序排列、元素互不相同的整数数组 nums 和目标值 target，
      如果 target 存在就返回它的下标；否则返回它按顺序插入后应有的下标。
      要求时间复杂度 O(log n)。例如 nums = [1, 3, 5, 6]，target = 5 返回 2，
      target = 2 返回 1，target = 7 返回 4。

思路：插入位置就是「第一个大于等于 target 的元素的下标」，也叫 lower_bound。
      与 704 的标准二分略有不同：这里即便 nums[mid] 恰好等于 target，也不必立刻返回，
      因为我们要的是「最靠左」的那个满足 >= 的位置。
      把判定统一成「nums[mid] 是否小于 target」：
        - 是：整个左半段都小于 target，答案一定在右边，令 left = mid + 1；
        - 否：mid 本身可能是答案，但左边也许还有更靠前的，令 right = mid - 1 继续往左压。
      循环结束时 left > right，left 正好停在第一个 >= target 的位置：
      可能落在数组范围内（插入到某个元素之前），也可能等于 n（插到末尾）。

      为什么不用先判相等再各自处理：把「等于」和「大于」都归到同一个分支（不往右走），
      代码更短，也天然得到插入位置，这正是 lower_bound 的统一写法。

复杂度：时间 O(log n)，空间 O(1)。
"""


def search_insert(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


if __name__ == "__main__":
    assert search_insert([1, 3, 5, 6], 5) == 2
    assert search_insert([1, 3, 5, 6], 2) == 1
    assert search_insert([1, 3, 5, 6], 7) == 4
    assert search_insert([1, 3, 5, 6], 0) == 0
    assert search_insert([1], 0) == 0
    assert search_insert([1], 2) == 1
    assert search_insert([1], 1) == 0
    assert search_insert([], 5) == 0
    print("search_insert: all tests passed")
