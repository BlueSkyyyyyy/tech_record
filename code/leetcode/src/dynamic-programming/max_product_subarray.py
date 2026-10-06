"""152. 乘积最大子数组（Maximum Product Subarray）

题目：给定一个整数数组 nums，找出乘积最大的连续子数组（至少包含一个元素），
返回该子数组的乘积。

思路（动态规划：一个状态不够，就同时维护最大与最小）：
    先照搬 53「最大子数组和」的办法，设

        dp[i] = 以 nums[i] 结尾的最大乘积。

    可是这次会翻车：乘法的符号会翻转。一个很小的负数乘以另一个负数，反而变成
    很大的正数；如果只留「最大乘积」，就丢掉了这个潜力股。所以状态定义必须携带
    足够的信息，改成同时记录两端：

        cur_max = 以当前元素结尾的最大乘积
        cur_min = 以当前元素结尾的最小乘积（负得最厉害的那个，可能翻盘）

    读到 x 时，以 x 结尾的乘积只可能来自三种情况：从 x 重新开始、接上旧的
    cur_max、接上旧的 cur_min。三种一起比较，大的交给 cur_max、小的交给 cur_min：

        cur_max, cur_min = max(x, cur_max*x, cur_min*x), min(x, cur_max*x, cur_min*x)

    注意右边用的是同一轮更新前的旧值，所以要先算好三个候选再赋值（Python 元组
    同时赋值天然满足；C++ 需用临时变量中转）。全局最优 best 一路上取 cur_max。

    为什么负数不再是「拖后腿」：在求和问题里负数只会让和变小，所以 53 能放心地
    把负前缀丢掉；在乘积问题里负数可能在下一次乘法中「咸鱼翻身」，因此必须把
    最小值也一起留着——这正是「状态定义要覆盖所有会互相转化的量」的范例。

复杂度：时间 O(n)，空间 O(1)。
"""


def max_product(nums):
    cur_max = cur_min = best = nums[0]
    for x in nums[1:]:
        candidates = (x, cur_max * x, cur_min * x)
        cur_max = max(candidates)
        cur_min = min(candidates)
        best = max(best, cur_max)
    return best


if __name__ == "__main__":
    assert max_product([2, 3, -2, 4]) == 6
    assert max_product([-2, 0, -1]) == 0
    assert max_product([-2, 3, -4]) == 24
    assert max_product([0, 2]) == 2
    assert max_product([-2]) == -2
    assert max_product([1, -2, 3, -4]) == 24
    print("max_product_subarray: all tests passed")
