"""503. 下一个更大元素 II（Next Greater Element II）

题目：给定一个循环数组 nums（最后一个元素的下一个元素是数组的第一个元素），
      返回 nums 中每个元素的下一个更大元素；不存在则输出 -1。

思路（单调栈 + 把数组「接长一倍」）：
    循环数组的麻烦在于，一个元素的更大值可能出现在它左边。
    解决办法是逻辑上把数组遍历两遍：用下标 i 从 0 走到 2n-1，真实元素取 nums[i % n]。
    第一遍负责建立栈、结算第一遍里能确定的答案；第二遍让「绕回开头」的元素
    有机会去结算那些一直没等到更大值的元素。

    关键细节：只有 i < n 时才把下标压栈。第二遍是「补算」用的，
    不能再往里塞重复下标，否则同一位置会被处理两次，还会让栈无限增长。

    为什么遍历两圈就够：任意元素的下一个更大值，要么在它右侧（第一圈就能找到），
    要么在它左侧、需要绕一圈（第二圈就能覆盖）。两圈之后仍没结算的，就是真的没有更大值，
    保留初始的 -1。

复杂度：时间 O(n)（每个下标最多进出栈一次），空间 O(n)。
"""


def next_greater_elements(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(2 * n):
        x = nums[i % n]
        while stack and nums[stack[-1]] < x:
            result[stack.pop()] = x
        if i < n:
            stack.append(i)
    return result


if __name__ == "__main__":
    assert next_greater_elements([1, 2, 1]) == [2, -1, 2]
    assert next_greater_elements([1, 2, 3, 4, 3]) == [2, 3, 4, -1, 4]
    assert next_greater_elements([5, 4, 3, 2, 1]) == [-1, 5, 5, 5, 5]
    print("next_greater_elements_ii: all tests passed")
