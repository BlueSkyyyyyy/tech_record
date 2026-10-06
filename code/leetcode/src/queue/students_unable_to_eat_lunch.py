"""1700. 无法吃午餐的学生数量（Number of Students Unable to Eat Lunch）

题目：学生排队，sandwiches 是栈（sandwiches[0] 是栈顶）。队首学生若喜欢栈顶的
三明治就取走并离开，否则走到队尾。问到不能有人再取走时，还有多少学生吃不上。

思路（队列模拟 → 计数模拟）：
    直接在队列里一轮轮转也能做，但有一个更省的事实：学生一旦「不喜欢」就回队尾，
    他们的相对顺序不影响结果，真正决定谁能吃上的只有**喜欢 0 和喜欢 1 各有多少人**。

    于是只需数出 0、1 的学生数量，然后按栈顶顺序发三明治：
    - 栈顶是 0 且还有喜欢 0 的学生，就发出去，喜欢 0 的计数减一；
    - 栈顶是 1 同理；
    - 一旦栈顶的那种三明治已经没人要了，后面也不可能再发出去，直接停下。
    剩下的学生数就是答案。

复杂度：时间 O(n + m)，空间 O(1)。
"""


def count_students(students, sandwiches):
    zeros = students.count(0)
    ones = len(students) - zeros
    for s in sandwiches:
        if s == 0 and zeros > 0:
            zeros -= 1
        elif s == 1 and ones > 0:
            ones -= 1
        else:
            break
    return zeros + ones


if __name__ == "__main__":
    assert count_students([1, 1, 0, 0], [0, 1, 0, 1]) == 0
    assert count_students([1, 1, 1, 0, 0, 1], [1, 0, 0, 0, 1, 1]) == 3
    assert count_students([0], [1]) == 1
    assert count_students([1], [1]) == 0
    print("count_students: all tests passed")
