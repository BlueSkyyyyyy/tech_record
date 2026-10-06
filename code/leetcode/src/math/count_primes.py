"""204. 计数质数（Count Primes）

题目：给定整数 n，返回所有小于 n 的质数的数量。

思路（埃拉托斯特尼筛）：
    开一个长度为 n 的布尔数组，初始假定每个数都是质数。i 从 2 开始：如果 i 仍是质数，
    就把 i 的所有倍数（从 i * i 开始）标记为合数。最后统计未被标记的个数。

    为什么每个合数都会被筛掉：设合数 m = a * b（a, b > 1），则 m 一定是某个小于它的
    质数 p（取 m 的最小质因子）的倍数，而当外层走到 p 时，p 的所有倍数都会被筛，m 自然
    被覆盖。

    为什么倍数从 i * i 而不是 i * 2 开始：小于 i * i 的、i 的倍数，比如 i * k
    （k < i），已经在更早的轮次被 k 的最小质因子筛掉了，从 i * i 开始才不重复劳动。

    为什么外层只需到 sqrt(n)：若 i 是合数，它必有不超过 sqrt(i) 的质因子，那个质因子
    在更早的轮次已经把它筛掉了。所以外层循环写 `while i * i < n` 即可，剩下的没被筛的
    都是质数。

复杂度：时间 O(n log log n)，空间 O(n)。
"""


def count_primes(n):
    if n < 3:
        return 0
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    i = 2
    while i * i < n:
        if is_prime[i]:
            for j in range(i * i, n, i):
                is_prime[j] = False
        i += 1
    return sum(is_prime)


if __name__ == "__main__":
    assert count_primes(0) == 0
    assert count_primes(1) == 0
    assert count_primes(2) == 0
    assert count_primes(3) == 1
    assert count_primes(10) == 4
    assert count_primes(11) == 4
    assert count_primes(30) == 10
    assert count_primes(100) == 25
    print("count_primes: all tests passed")
