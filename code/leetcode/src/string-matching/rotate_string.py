"""796. 旋转字符串（Rotate String）

题目：给定字符串 s 和 goal，判断 s 能否经过若干次「把最左边字符移到最右边」的操作
变成 goal。

思路（s + s 判包含）：
    s 的所有旋转结果，恰好是 s + s 里全部长度为 n 的窗口：
        s = "abcde"，则 s + s = "abcdeabcde"，
        从每个下标取长度 5 的窗口就得到 eabcde / deabcd / ... 五种旋转。
    所以只要先比长度，再看 goal 是不是 (s + s) 的子串即可。

复杂度：时间 O(n)（标准子串查找），空间 O(n)（拼接出的 s + s）。
"""


def rotate_string(s, goal):
    return len(s) == len(goal) and goal in s + s


if __name__ == "__main__":
    assert rotate_string("abcde", "cdeab") is True
    assert rotate_string("abcde", "abced") is False
    assert rotate_string("a", "a") is True
    assert rotate_string("", "") is True
    assert rotate_string("abc", "abcd") is False
    print("rotate_string: all tests passed")
