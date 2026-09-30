"""739. 每日温度（Daily Temperatures）

题目：给定一个整数数组 temperatures，返回一个数组 answer，其中 answer[i]
      表示第 i 天之后要等多少天才会出现一个更高的温度；若之后都不会更高，则填 0。

思路（单调递减栈，栈里存下标）：
    从左到右扫描，维护一个「温度单调递减」的下标栈。
    每遇到当天温度 t，只要栈顶那天的温度比 t 低，就说明栈顶那天「等到了」更高的温度：
    答案就是 i - 栈顶下标，然后弹掉它。重复直到栈空或栈顶温度 >= t，再把当天压栈。

    为什么用栈而不是两层循环：暴力对每天向后找，最坏 O(n^2)。
    但注意一个事实——每天只需要找到「右边第一个更高的温度」。当我们从左往右扫时，
    栈里保存的是「还没等到更高温度」的那些天，它们的温度从左到右递减。
    新来的一天如果能满足栈顶，就一定能满足栈里更靠内的若干天（因为那些天更冷、且更靠左），
    于是可以一次性弹出结算，每个下标最多进栈出栈一次，整体 O(n)。

    为什么栈里存下标不存温度：答案要的是「相差多少天」，必须知道位置；存下标即可同时取温度。

复杂度：时间 O(n)（每个下标进出栈一次），空间 O(n)。
"""


def daily_temperatures(temperatures):
    n = len(temperatures)
    answer = [0] * n
    stack = []
    for i, t in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < t:
            j = stack.pop()
            answer[j] = i - j
        stack.append(i)
    return answer


if __name__ == "__main__":
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
    assert daily_temperatures([30, 60, 90]) == [1, 1, 0]
    assert daily_temperatures([90, 80, 70]) == [0, 0, 0]
    print("daily_temperatures: all tests passed")
