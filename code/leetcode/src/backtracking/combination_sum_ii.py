"""40. 组合总和 II（Combination Sum II）

题目：给定一个候选人编号的集合 candidates 和一个目标数 target，找出所有
可以使数字和为 target 的组合。candidates 中的每个数字在每个组合中只能
使用一次。注意解集中不能包含重复的组合。

思路（排序 + start 传 i+1 + 同层去重）：
    这道题是 39 与 90 的合体：
      - 每个数只能用一次 → 递归传 i + 1（和 39 的传 i 相反）；
      - 候选数组可能含重复且组合不能重复 → 排序后用同层去重，规则与 90 一致：
        i > start 且 candidates[i] == candidates[i-1] 时跳过。

    为什么要同时具备这两点：传 i + 1 只保证「下标不回头」，挡不住「值相同、
    下标不同」造成的重复组合；同层去重补上这一刀。反过来，只去重不传 i + 1
    会把同一个元素用多次，也不对。两把锁缺一不可。

复杂度：时间与答案规模相关，最坏指数级；空间 O(n)（递归深度，不含答案本身）。
"""


def combination_sum2(candidates, target):
    candidates = sorted(candidates)
    res = []
    path = []

    def backtrack(start, remain):
        if remain == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remain:
                break
            if i > start and candidates[i] == candidates[i - 1]:
                continue
            path.append(candidates[i])
            backtrack(i + 1, remain - candidates[i])
            path.pop()

    backtrack(0, target)
    return res


if __name__ == "__main__":
    out = combination_sum2([10, 1, 2, 7, 6, 1, 5], 8)
    want = [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
    assert out == want

    assert combination_sum2([2, 5, 2, 1, 2], 5) == [[1, 2, 2], [5]]
    assert combination_sum2([1], 2) == []
    print("combination_sum2: all tests passed")
