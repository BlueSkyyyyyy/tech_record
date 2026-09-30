"""202. 快乐数（Happy Number）

题目：判断一个正整数 n 是否为「快乐数」。快乐数的定义是：把 n 替换成它各位数字的平方和，
反复进行，如果最终能得到 1，就是快乐数；如果陷入不包含 1 的循环，就不是。
例如 19 -> 1^2+9^2=82 -> 68 -> 100 -> 1，所以 19 是快乐数。

思路：把「反复求平方和」看成一串状态转移。要么最终到达 1，要么进入一个循环。
     「判断会不会重复」正是哈希集合的强项：
       - 每一步算出下一个数 next；
       - 若 next == 1，返回 True；
       - 若 next 已经在集合里，说明进入了循环，返回 False；
       - 否则把 next 记进集合，继续。
     为什么需要集合：不记已访问的数，循环会无限进行下去；集合保证每个状态最多访问一次，
     因此总步数有限（数学上这个迭代很快收敛）。

复杂度：时间 O(log n)（每次迭代把数字位数减少，循环长度有上界），空间 O(log n)。
"""


def is_happy(n):
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return n == 1


if __name__ == "__main__":
    assert is_happy(19) is True
    assert is_happy(1) is True
    assert is_happy(2) is False
    assert is_happy(7) is True
    assert is_happy(100) is True
    print("happy_number: all tests passed")
