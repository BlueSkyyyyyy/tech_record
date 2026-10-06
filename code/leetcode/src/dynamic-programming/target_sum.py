"""494. 目标和（Target Sum）

题目：给你一个非负整数数组 nums 和一个整数 target，向数组中的每个整数前添加 '+'
或 '-'，然后串联所有整数，返回可以通过上述方法构造出运算结果等于 target 的不同
表达式的数目。

思路（化为 0/1 背包·数方案）：
    直接枚举每个数取正或取负是指数级的。把式子改写一下：设取 '+' 的数之和为 P，取
    '-' 的数之和为 N（都是非负），则

        P - N = target,   P + N = total（数组总和）

    两式相加得 P = (total + target) / 2。也就是说：**给哪些数加正号，等价于从数组中
    选出一个和为 P 的子集**。于是问题变成「选出和为 P 的子集有多少种选法」。

    设 dp[i] 表示选出的数之和为 i 的方案数，就是标准 0/1 背包数方案：

        dp[i] += dp[i - num]   （i 从 P 递减到 num）

    初始化 dp[0] = 1（什么都不选是一种方案）。答案 dp[P]。

    可行性先判掉：若 (total + target) 为奇数（P 不是整数），或 |target| > total
    （正负号怎么摆都够不到），直接返回 0。

    为什么能这样转化：把「符号选择」翻译成「子集选择」，是因为加号集合一旦确定，
    其余全是减号，整个表达式就唯一确定了。一次转化省掉了指数枚举，是这类题最漂亮
    的一步。

复杂度：时间 O(n × P)，空间 O(P)，其中 P = (total + target) / 2。
"""


def find_target_sum_ways(nums, target):
    total = sum(nums)
    if abs(target) > total or (total + target) % 2 != 0:
        return 0
    p = (total + target) // 2
    dp = [0] * (p + 1)
    dp[0] = 1
    for num in nums:
        for i in range(p, num - 1, -1):
            dp[i] += dp[i - num]
    return dp[p]


if __name__ == "__main__":
    assert find_target_sum_ways([1, 1, 1, 1, 1], 3) == 5
    assert find_target_sum_ways([1], 1) == 1
    assert find_target_sum_ways([1, 2, 3], 0) == 2
    assert find_target_sum_ways([1, 2], 3) == 1
    assert find_target_sum_ways([1, 0], 1) == 2
    assert find_target_sum_ways([1, 2], 5) == 0
    print("target_sum: all tests passed")
