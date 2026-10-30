"""887. 鸡蛋掉落（Super Egg Drop）

题目：k 枚鸡蛋、n 层楼。存在一个临界楼层 f：从 <= f 层扔下鸡蛋不会碎，从 > f 层扔会碎。
每次可以选一层扔一枚鸡蛋，鸡蛋碎了就不能再用。求**保证**能确定 f 的最少实验次数。

思路（DP + 二分优化转移）：
    朴素状态：dp[i][j] = 用 i 枚鸡蛋、确定 j 层所需的最少次数。枚举第一次扔的楼层 x：
        碎了 → 剩 i - 1 枚蛋，答案在下面 x - 1 层：dp[i-1][x-1]
        没碎 → 还有 i 枚蛋，答案在上面 j - x 层：dp[i][j-x]
        dp[i][j] = 1 + min_x max(dp[i-1][x-1], dp[i][j-x])
    直接枚举 x 是 O(k n²)。注意两项关于 x 一个是「随 x 递增」，一个是「随 x 递减」，
    它们的 max 是单峰的，于是可以二分找两线的交点，把内层降到 O(log n)。

    二分的判定：若 dp[i-1][x-1] > dp[i][j-x]，说明「碎了」那一支更贵，x 该往小走；
    否则往大走。取 min 时把每次二分到的候选都试一遍即可。

复杂度：时间 O(k · n log n)，空间 O(k · n)（可滚动到 O(n)，这里保留两维便于阅读）。
"""


def super_egg_drop(k, n):
    dp = [[0] * (n + 1) for _ in range(k + 1)]
    for floors in range(1, n + 1):
        dp[0][floors] = float("inf")  # 没有蛋就确定不了任何楼层
    for eggs in range(1, k + 1):
        for floors in range(1, n + 1):
            lo, hi = 1, floors
            best = float("inf")
            while lo <= hi:
                mid = (lo + hi) // 2
                broken = dp[eggs - 1][mid - 1]
                intact = dp[eggs][floors - mid]
                best = min(best, 1 + max(broken, intact))
                if broken > intact:
                    hi = mid - 1
                else:
                    lo = mid + 1
            dp[eggs][floors] = best
    return dp[k][n]


if __name__ == "__main__":
    assert super_egg_drop(1, 2) == 2
    assert super_egg_drop(2, 6) == 3
    assert super_egg_drop(3, 14) == 4
    assert super_egg_drop(2, 1) == 1
    assert super_egg_drop(1, 10) == 10
    print("super_egg_drop: all tests passed")
