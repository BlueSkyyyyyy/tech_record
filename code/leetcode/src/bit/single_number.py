"""136. 只出现一次的数字（Single Number）

题目：给定一个非空整数数组 nums，除了某个元素只出现一次以外，其余每个元素都出现
两次。找出那个只出现一次的元素。要求线性时间、常数额外空间。

思路（异或消消乐）：
    把数组里所有数依次异或起来，剩下的就是那个只出现一次的数。异或有三个性质：
      - 交换律：a ^ b == b ^ a，可以随意调整顺序；
      - 结合律：(a ^ b) ^ c == a ^ (b ^ c)；
      - 自反性：a ^ a == 0，且 a ^ 0 == a。
    成对出现的数两两抵消（a ^ a = 0），只有落单的那个没被抵消，最终结果就是它。

    为什么用异或而不是哈希表：哈希表需要 O(n) 额外空间来记录每个数出现过几次，
    而异或把所有元素「压缩」成一个累加器，天然满足「两两抵消」的语义，额外空间是
    O(1)。这类「配对抵消 / 找落单」的题，异或几乎是最优解。

    推广到「其余元素出现 k 次」时，逐个二进制位统计 1 的个数、对 k 取模即可；而
    本题 k = 2 恰好等价于异或（不进位的二进制加法）。

复杂度：时间 O(n)（每个元素异或一次），空间 O(1)。
"""


def single_number(nums):
    result = 0
    for x in nums:
        result ^= x
    return result


if __name__ == "__main__":
    assert single_number([2, 2, 1]) == 1
    assert single_number([4, 1, 2, 1, 2]) == 4
    assert single_number([1]) == 1
    assert single_number([-1, -1, -2]) == -2
    assert single_number([0, 0, 5]) == 5
    assert single_number([7, 3, 7]) == 3
    print("single_number: all tests passed")
