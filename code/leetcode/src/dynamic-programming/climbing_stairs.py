"""70. 爬楼梯（Climbing Stairs）

题目：假设你正在爬楼梯，需要 n 阶才能到达楼顶。每次你可以爬 1 或 2 个台阶，
有多少种不同的方法可以爬到楼顶？

思路（动态规划：数方案用加法）：
    设 dp[i] 表示爬到第 i 阶的方法数。想到达第 i 阶，最后一步只可能来自
    第 i-1 阶（迈 1 步）或第 i-2 阶（迈 2 步）。这两类走法互不重复，
    所以「到 i 的方案数」=「到 i-1 的方案数」+「到 i-2 的方案数」：

        dp[i] = dp[i-1] + dp[i-2]

    初始化：dp[0] = 1（站在地面算一种「还没走」的状态），dp[1] = 1（只能迈 1 步）。
    之后从小到大递推即可。

    这道题的递推式和 509 斐波那契完全一样（初始项不同），可以对照记忆：
    斐波那契从 0、1 起步，爬楼梯从 1、1 起步。因为只依赖前两项，同样可以
    用两个滚动变量把空间压到 O(1)。

    为什么想到用 DP 而不是暴力枚举：暴力要枚举所有 1/2 的排列，是指数级；
    而「到了第几阶」这一维状态一旦确定，后面的走法数就固定了，正是重复子问题，
    适合用一张递推表/几个变量自底向上累积。

复杂度：时间 O(n)，空间 O(1)。
"""


def climb_stairs(n):
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1


if __name__ == "__main__":
    assert climb_stairs(1) == 1
    assert climb_stairs(2) == 2
    assert climb_stairs(3) == 3
    assert climb_stairs(4) == 5
    assert climb_stairs(5) == 8
    assert climb_stairs(10) == 89
    print("climbing_stairs: all tests passed")
