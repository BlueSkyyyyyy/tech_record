"""1524. 和为奇数的子数组数目（Number of Sub-arrays With Odd Sum）

题目：给定整数数组 arr，返回元素之和为奇数的（连续、非空）子数组的数目。
      答案可能很大，对 10^9 + 7 取模。例如 arr = [1, 3, 5] 返回 4，
      arr = [2, 4, 6] 返回 0，arr = [1, 2, 3, 4, 5, 6, 7] 返回 16。

思路：子数组 arr[j..i-1] 的和为 prefix[i] - prefix[j]。一个整数减法结果是不是奇数，
      只取决于两个操作数的奇偶性：一奇一偶相减得奇数，同奇同偶得偶数。
      所以「子数组和为奇数」等价于「两个前缀和的奇偶性不同」。
      于是每遇到一对奇偶性不同的前缀，就贡献一个合法子数组；把所有这样的配对数加起来即可。
      用两个计数器分别数出前缀和里偶数、奇数的个数，答案就是 even * odd
      （每一个偶前缀配每一个奇前缀，顺序天然对应一个非空子数组）。
      前缀和从「空前缀 prefix[0] = 0」开始，它算一个偶前缀，所以 even 初值为 1。
      注意这里不需要哈希表：配对条件是「奇偶不同」，只有两类，数出两类数量相乘即可 O(1) 出答案，
      这正是「同余前缀和」在模为 2 时的特例（余数只有 0 和 1 两种）。

复杂度：时间 O(n)，空间 O(1)。
"""

MOD = 10**9 + 7


def num_of_subarrays_with_odd_sum(arr):
    even = 1
    odd = 0
    prefix = 0
    for x in arr:
        prefix += x
        if prefix % 2:
            odd += 1
        else:
            even += 1
    return (even * odd) % MOD


if __name__ == "__main__":
    assert num_of_subarrays_with_odd_sum([1, 3, 5]) == 4
    assert num_of_subarrays_with_odd_sum([2, 4, 6]) == 0
    assert num_of_subarrays_with_odd_sum([1, 2, 3, 4, 5, 6, 7]) == 16
    assert num_of_subarrays_with_odd_sum([1]) == 1
    assert num_of_subarrays_with_odd_sum([100]) == 0
    assert num_of_subarrays_with_odd_sum([1, 1]) == 2
    assert num_of_subarrays_with_odd_sum([1, 2]) == 2
    print("subarrays_with_odd_sum: all tests passed")
