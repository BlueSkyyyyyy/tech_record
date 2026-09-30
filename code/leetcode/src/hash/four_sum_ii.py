"""454. 四数相加 II（4Sum II）

题目：给定四个等长整数数组 A、B、C、D，统计有多少个四元组 (i, j, k, l) 满足
A[i] + B[j] + C[k] + D[l] == 0。

思路：四层循环是 O(n^4)，太慢。把四个数分成两半，各算「两两之和」：
       - 先枚举 A、B 的所有组合，把和以及它出现的次数记进哈希表；
       - 再枚举 C、D 的所有组合，和为 s 时，查表里有多少个 -s，
         这些组合都能和当前 (k, l) 配成 0。
     于是复杂度从 O(n^4) 降到 O(n^2)。
     为什么存「次数」而不是只存集合：题目要统计**方案数**，不同的 (i, j) 即使和相同
     也是不同方案，所以必须记录每个和出现了几次。
     这是「用哈希表把两次枚举的乘积」这一思想的典型题。

复杂度：时间 O(n^2)，空间 O(n^2)。
"""


def four_sum_count(a, b, c, d):
    ab = {}
    for x in a:
        for y in b:
            ab[x + y] = ab.get(x + y, 0) + 1
    total = 0
    for x in c:
        for y in d:
            total += ab.get(-(x + y), 0)
    return total


if __name__ == "__main__":
    assert four_sum_count([1, 2], [-2, -1], [-1, 2], [0, 2]) == 2
    assert four_sum_count([0], [0], [0], [0]) == 1
    assert four_sum_count([1], [1], [1], [1]) == 0
    print("four_sum_ii: all tests passed")
