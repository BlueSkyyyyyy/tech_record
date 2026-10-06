"""1601. 最多可达成的换楼请求数目（Maximum Number of Achievable
Transfer Requests）

题目：有 n 栋楼，requests[i] = [from, to] 表示一次"从 from 搬到 to"的请求。选出
尽可能多的请求同时满足，使得对每栋楼而言"搬出人数 == 搬入人数"。求最多能同时满足
多少条请求。

思路（枚举请求子集，看净变化是否全零）：
    请求之间没有先后依赖，一个请求要么满足要么不满足，共 2^R 种选法（R ≤ 16）。
    对每一种子集，统计每栋楼的净变化 delta：搬出 -1、搬入 +1。这个子集可行，当且仅当
    所有楼的净变化都为 0（人不会凭空多出或消失）。

    于是直接枚举子集，维护最大可行请求数即可。枚举时用掩码的第 i 位表示"第 i 条请求
    被选中"，是最朴素的"位掩码当开关表"。

复杂度：时间 O(2^R * R + 2^R * n)，空间 O(n)（每轮重置 delta 数组）。
"""


def maximum_requests(n, requests):
    best = 0
    for mask in range(1 << len(requests)):
        delta = [0] * n
        count = 0
        for i, (frm, to) in enumerate(requests):
            if mask >> i & 1:
                delta[frm] -= 1
                delta[to] += 1
                count += 1
        if count > best and all(d == 0 for d in delta):
            best = count
    return best


if __name__ == "__main__":
    assert maximum_requests(5, [[0, 1], [1, 0], [0, 1], [1, 2], [2, 0], [3, 4]]) == 5
    assert maximum_requests(3, [[0, 0], [1, 2], [2, 1]]) == 3
    assert maximum_requests(4, [[0, 3], [3, 1], [1, 2], [2, 0]]) == 4
    assert maximum_requests(2, [[0, 1]]) == 0
    print("maximum_requests: all tests passed")
