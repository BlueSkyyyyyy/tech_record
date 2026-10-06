"""179. 最大数（Largest Number）

题目：给你一个非负整数数组 nums，把它们重新排列后拼成一个数，要求拼出的数最大，
      以字符串形式返回（结果可能很大）。

思路（自定义比较器，比的是「谁排前面更划算」）：
    直觉上是不是把大的数字放前面就行？不是。比如 3 和 30，数字上 3 小，但拼成
    "330" 比 "303" 大，所以 3 应该排在 30 前面。这说明排序依据不是数值大小，而是
    「两个数谁放前面能拼出更大的结果」。

    对任意两个数 a、b，比较拼法 `a+b` 与 `b+a`（按字符串拼接后逐字符比较字典序，
    等长时字典序大小与数值大小一致）：若 `a+b > b+a`，就规定 a 排在 b 前面。这个
    关系满足全序，可以用它当比较器对整个数字串排序，再把排好的串依次拼起来。

    为什么这个比较器对「整体最优」有效：拼接结果可以看成所有数字串按某个顺序首尾
    相接。若存在相邻两项 a、b 使 `a+b < b+a`，交换它们会让整体结果变大（其余部分
    不变），所以最优排列里不存在这样的逆序对。于是「任意相邻都满足 a+b >= b+a」的
    排列就是最优，这正是按上述比较器排序得到的结果（交换论证）。

    最后要处理前导零：如果排完第一个字符是 '0'，说明所有数都是 0，直接返回 "0"。
"""


def largest_number(nums):
    from functools import cmp_to_key

    def cmp(a, b):
        if a + b > b + a:
            return -1
        if a + b < b + a:
            return 1
        return 0

    strs = [str(x) for x in nums]
    strs.sort(key=cmp_to_key(cmp))
    res = "".join(strs)
    return "0" if res[0] == "0" else res


if __name__ == "__main__":
    assert largest_number([10, 2]) == "210"
    assert largest_number([3, 30, 34, 5, 9]) == "9534330"
    assert largest_number([0, 0]) == "0"
    assert largest_number([1]) == "1"
    assert largest_number([3, 30]) == "330"
    print("largest_number: all tests passed")
