"""69. x 的平方根（Sqrt(x)）

题目：给定一个非负整数 x，计算并返回它的算术平方根的整数部分（向下取整）。
      不允许使用任何内置指数函数和算符。例如 x = 4 返回 2，x = 8 返回 2（因为 sqrt(8) ≈ 2.828，下取整为 2）。

思路：本题要的不是「在有序数组里找一个值」，而是「找一个数 k，使 k*k <= x 且 k 尽可能大」。
      这类问题叫**二分答案**：答案本身落在某个区间里，且判定「某个数行不行」的条件具有单调性——
      如果 k 行（k*k <= x），那么所有比 k 小的数都行；如果 k 不行，那么比 k 大的数也都不行。
      有了单调性，就可以对「答案的值域」做二分，而不是对数组下标做二分。

      维护闭区间 [left, right]，它表示「答案 k 只可能落在这里」。每轮取 mid：
        - mid*mid <= x：mid 可行，但也许还有更大的可行值，令 left = mid + 1；
        - mid*mid > x：mid 太大，令 right = mid - 1。
      循环结束（left > right）时，right 恰好是最后一个满足 mid*mid <= x 的数，也就是 floor(sqrt(x))。

      为什么右边界可以取 x 而不是 x // 2：x 很小时（x=0/1）x//2 会小于真实答案；
      统一取 right = x 更稳妥，多出的对数级次数可以忽略。

      为什么 C++ 里 mid*mid 要用 long long：mid 最大接近 2^31，平方会溢出 32 位 int。
      C++ 实现里用 1LL * mid * mid 先提升到 64 位。

复杂度：时间 O(log x)，空间 O(1)。
"""


def my_sqrt(x):
    left, right = 0, x
    while left <= right:
        mid = left + (right - left) // 2
        if mid * mid <= x:
            left = mid + 1
        else:
            right = mid - 1
    return left - 1


if __name__ == "__main__":
    assert my_sqrt(0) == 0
    assert my_sqrt(1) == 1
    assert my_sqrt(4) == 2
    assert my_sqrt(8) == 2
    assert my_sqrt(9) == 3
    assert my_sqrt(15) == 3
    assert my_sqrt(16) == 4
    assert my_sqrt(2147395599) == 46339
    print("sqrtx: all tests passed")
