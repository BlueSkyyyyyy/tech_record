"""1395. 统计作战单位数（Count Number of Teams）

题目：n 个士兵排成一排，每个士兵有一个唯一的能力值 rating[i]。选出 3 个下标
    i < j < k，若 rating[i] < rating[j] < rating[k] 或 rating[i] > rating[j] > rating[k]，
    则组成一个作战单位。求作战单位的总数。

思路（枚举中间人 + 乘法原理 + 值域树状数组）：
    三个数的单调关系可以「以中间那个 j 为中心」拆开：
      - 递增队形：左边比 j 小的个数 × 右边比 j 大的个数；
      - 递减队形：左边比 j 大的个数 × 右边比 j 小的个数。
    对每个 j 把两类乘积累加即可。

    怎么快速得到这四个数？从左往右扫，用值域树状数组维护「左侧已出现元素」：
      left_less  = prefix(rating[j] - 1)；
      left_greater = left_count - prefix(rating[j])。
    右侧的个数用「全局个数 - 左侧个数」推出来，避免再扫一遍：
      - 全局严格小于 rating[j] 的个数用一次计数前缀求得；
      - 全局严格大于 rating[j] 的个数同理。

    这就是「前后各数一遍、用乘法原理合并」的典型套路；两遍扫描也可以，但用全局计数
    更省代码。

复杂度：时间 O(n log M + n + M)（M = 值域上界），空间 O(M)。
"""


class Fenwick:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s


def num_teams(rating):
    maxv = max(rating)
    total = [0] * (maxv + 2)
    for x in rating:
        total[x] += 1

    less_total = [0] * (maxv + 2)
    greater_total = [0] * (maxv + 2)
    run = 0
    for v in range(1, maxv + 1):
        less_total[v] = run
        run += total[v]
    run = 0
    for v in range(maxv, 0, -1):
        greater_total[v] = run
        run += total[v]

    bit = Fenwick(maxv)
    ans = 0
    left_count = 0
    for x in rating:
        left_less = bit.prefix(x - 1)
        left_greater = left_count - bit.prefix(x)
        right_less = less_total[x] - left_less
        right_greater = greater_total[x] - left_greater
        ans += left_less * right_greater + left_greater * right_less
        bit.add(x, 1)
        left_count += 1
    return ans


if __name__ == "__main__":
    assert num_teams([2, 5, 3, 4, 1]) == 3
    assert num_teams([2, 1, 3]) == 0
    assert num_teams([1, 2, 3, 4]) == 4
    assert num_teams([4, 3, 2, 1]) == 4
    assert num_teams([1, 3, 2, 4]) == 2

    print("num_teams: all tests passed")
