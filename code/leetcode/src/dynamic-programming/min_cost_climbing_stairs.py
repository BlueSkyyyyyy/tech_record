"""746. 使用最小花费爬楼梯（Min Cost Climbing Stairs）

题目：给你一个整数数组 cost，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。
一旦你支付此费用，即可选择向上爬一个或者两个台阶。你可以从下标 0 或下标 1 的台阶
开始爬。请你计算并返回达到楼梯顶部的最低花费（顶部在最后一个台阶之上）。

思路（动态规划：带权的最短路径，取 min）：
    设 dp[i] 表示「到达第 i 个台阶并支付了它」的最小累计花费。
    想到第 i 个台阶，上一步只能来自第 i-1 或第 i-2 个台阶，于是：

        dp[i] = min(dp[i-1], dp[i-2]) + cost[i]

    初始化：可以从 0 或 1 出发，dp[0] = cost[0]，dp[1] = cost[1]。
    最后到达顶部时，可以从最后一个台阶（i = n-1）迈一步上去，也可以从倒数第二个
    台阶（i = n-2）迈两步直接上去，所以答案是 min(dp[n-1], dp[n-2])。

    和爬楼梯（70）对比：70 是「数方案数」所以用加法，746 是「求最小花费」所以用
    min。两者的状态和决策完全同构，只是合并左右两个来源时用了不同的运算符——
    这是「同一骨架、不同聚合」的典型。

    同样只依赖前两项，用两个滚动变量把空间压到 O(1)。

复杂度：时间 O(n)，空间 O(1)。
"""


def min_cost_climbing_stairs(cost):
    n = len(cost)
    if n <= 1:
        return 0
    prev2, prev1 = cost[0], cost[1]
    for i in range(2, n):
        prev2, prev1 = prev1, min(prev2, prev1) + cost[i]
    return min(prev2, prev1)


if __name__ == "__main__":
    assert min_cost_climbing_stairs([10, 15, 20]) == 15
    assert min_cost_climbing_stairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1]) == 6
    assert min_cost_climbing_stairs([0, 0, 0, 0]) == 0
    assert min_cost_climbing_stairs([5, 3]) == 3
    print("min_cost_climbing_stairs: all tests passed")
