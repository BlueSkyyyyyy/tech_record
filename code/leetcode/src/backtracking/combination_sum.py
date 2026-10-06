"""39. 组合总和（Combination Sum）

题目：给你一个无重复元素的整数数组 candidates 和一个目标整数 target，
找出 candidates 中可以使数字和为目标数 target 的所有不同组合。
candidates 中的同一个数字可以无限制重复被选取。

思路（回溯 + start，下一层仍从 i 开始）：
    组合问题的老规矩：用 start 保证只往后选，避免 [2,3] 与 [3,2] 重复。
    与 77 / 78 不同的只有一点——**同一个数可以重复选**，所以递归时传的是
    `i` 而不是 `i + 1`：本层选了 candidates[i] 之后，下一层仍然允许再选它。

    递归参数携带「还差多少」remain：remain == 0 时收答案；一旦某数已经比
    remain 大，后面的数只会更大（数组有序），直接跳过或 break。

    为什么传 i 就能表达「可重复」：`i` 表示下一层的起点仍是当前数，
    于是路径里可以连续出现多个相同的数；而起点不回头，又保证组合不重复。
    这两种约束正好是「可重复选取」和「组合不重复」的分工。

复杂度：时间与答案规模相关，最坏指数级；空间 O(target / min(candidates))
    （递归深度，不含答案本身）。
"""


def combination_sum(candidates, target):
    res = []
    path = []

    def backtrack(start, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain:
                break
            path.append(candidates[i])
            backtrack(i, remain - candidates[i])
            path.pop()

    backtrack(0, target)
    return res


if __name__ == "__main__":
    out = combination_sum([2, 3, 6, 7], 7)
    assert [2, 2, 3] in out
    assert [7] in out
    assert len(out) == 2

    assert combination_sum([2, 3, 5], 8) == [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
    assert combination_sum([2], 1) == []
    print("combination_sum: all tests passed")
