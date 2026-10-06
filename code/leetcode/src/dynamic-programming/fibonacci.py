"""509. 斐波那契数（Fibonacci Number）

题目：斐波那契数 F(0) = 0，F(1) = 1，之后 F(n) = F(n-1) + F(n-2)。
给定 n，计算 F(n)。

思路（动态规划：从「重复子问题」到递推表）：
    朴素递归 f(n) = f(n-1) + f(n-2) 会大量重复计算同一个 f(k)，是指数级。
    注意到每个数只由前两个数决定，于是自底向上把 F(0)、F(1) 算起，
    一路递推到 F(n)：每个新值只用它的前两项，算过就不会再算。

    这就是动态规划最小、最干净的形态：
      1. 状态定义：dp[i] 表示第 i 个斐波那契数；
      2. 转移方程：dp[i] = dp[i-1] + dp[i-2]；
      3. 初始化：dp[0] = 0，dp[1] = 1；
      4. 遍历顺序：i 从小到大，保证用到的值都已算好。

    又因为 dp[i] 只依赖前两项，其实不需要整张表，用两个滚动变量
    （prev2 存 dp[i-2]，prev1 存 dp[i-1]）即可把空间压到 O(1)。

    为什么不写递归 + 记忆化：那版同样正确、也好理解，但自底向上没有递归栈开销，
    常数更小；记忆化更适合「依赖关系不规则」的场景，本题依赖是一条直线。

复杂度：时间 O(n)（每个状态算一次），空间 O(1)（只用两个滚动变量）。
"""


def fib(n):
    if n < 2:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1


if __name__ == "__main__":
    assert fib(0) == 0
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(3) == 2
    assert fib(4) == 3
    assert fib(10) == 55
    assert fib(30) == 832040
    print("fibonacci: all tests passed")
