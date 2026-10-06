"""344. 反转字符串（Reverse String）

题目：给定一个字符数组 s，原地反转它，要求额外空间为 O(1)。

思路（对撞双指针）：
    用左右两个指针从两端向中间靠拢，每次交换它们指向的字符，然后 left 右移、right 左
    移，直到两指针相遇或交错。因为只交换、不再申请等长数组，所以额外空间是 O(1)。

    为什么必须原地：题目明确要求「修改输入数组、不额外分配」。Python 里字符串不可变，
    所以这里传入的是字符列表（list[str]），可以逐个换位；C++ 用 vector<char>& 直接改。

    这是字符串/数组题最基础的双指针模板：只要是从两头往中间处理、且元素可交换，都能套。
    后面的 541（分组反转）和 151（翻转单词）都是「反转」这个动作的组合。

复杂度：时间 O(n)（每个字符被访问一次），空间 O(1)。
"""


def reverse_string(s):
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1


if __name__ == "__main__":
    a = list("hello")
    reverse_string(a)
    assert a == list("olleh")

    b = list("Hannah")
    reverse_string(b)
    assert b == list("hannaH")

    c = list("a")
    reverse_string(c)
    assert c == list("a")

    d = []
    reverse_string(d)
    assert d == []

    e = list("ab")
    reverse_string(e)
    assert e == list("ba")

    print("reverse_string: all tests passed")
