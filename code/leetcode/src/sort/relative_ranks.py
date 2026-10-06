"""506. 相对名次（Relative Ranks）

题目：给你一个长度为 n 的数组 score，score[i] 是第 i 位运动员的成绩，所有成绩
      互不相同。返回一个答案数组 answer，其中 answer[i] 是第 i 位运动员的名次：
      前三名依次是 "Gold Medal"、"Silver Medal"、"Bronze Medal"，其余为名次数字的
      字符串形式（从 1 开始数，成绩最高者名次为 1）。

思路（排序 + 名次回填）：
    名次就是「按成绩从大到小排完后的位置」。但答案要按运动员原本的下标排列，所以
    不能直接返回排好序的数组，而要记住「每个位置对应第几名」。

    做法：先把下标数组按成绩降序排序，得到一个 order（order[0] 是成绩最高者的下标）。
    然后遍历 order，第 k 个位置对应的名次就是 k+1（k 从 0 开始），把这个名次写回
    answer[order[k]] 即可。

    为什么排下标而不是排名次对象：排序键是成绩，但我们最终要的是原下标处的答案，
    因此让下标跟着成绩一起排，排序后仍能通过 order 找回原位置。
"""


def find_relative_ranks(score):
    order = sorted(range(len(score)), key=lambda i: -score[i])
    medals = ["Gold Medal", "Silver Medal", "Bronze Medal"]
    res = [""] * len(score)
    for rank, i in enumerate(order):
        res[i] = medals[rank] if rank < 3 else str(rank + 1)
    return res


if __name__ == "__main__":
    assert find_relative_ranks([5, 4, 3, 2, 1]) == [
        "Gold Medal", "Silver Medal", "Bronze Medal", "4", "5",
    ]
    assert find_relative_ranks([10, 3, 8, 9, 4]) == [
        "Gold Medal", "5", "Bronze Medal", "Silver Medal", "4",
    ]
    assert find_relative_ranks([1]) == ["Gold Medal"]
    assert find_relative_ranks([3, 1, 2]) == ["Gold Medal", "Bronze Medal", "Silver Medal"]
    print("relative_ranks: all tests passed")
