"""1094. 拼车（Car Pooling）

题目：有一辆车，最多坐 capacity 人，只往一个方向开。给定行程 trips，每条是
      [numPassengers, from, to]，表示在 from 站上 numPassengers 人、在 to 站下车。
      判断这辆车能否把所有人从起点运到终点（任意时刻车上人数都不超过 capacity）。
      例如 trips = [[2,1,5],[3,3,7]]，capacity = 4，返回 False
      （第 3 到第 5 站之间车上同时有 2+3=5 人，超过 4）。

思路：把「站点」看成数组下标，乘客的上车/下车就是对区间 [from, to) 做一次「加 / 减」。
      这又是差分：每个行程做 diff[from] += num、diff[to] -= num，
      再从头累加 diff 得到每个站点区间内车上的人数，只要任意一处超过 capacity 就返回 False。
      为什么减在 to 而不是 to+1：to 是下车站，到达 to 时这些人已经离开，
      所以区间是左闭右开 [from, to)，在 to 处就把人数减掉。

复杂度：时间 O(S + m)（S 为最大站点编号，m 为行程数），空间 O(S)。
"""


def car_pooling(trips, capacity):
    size = 0
    for _, _, end in trips:
        size = max(size, end)
    diff = [0] * (size + 1)
    for num, start, end in trips:
        diff[start] += num
        diff[end] -= num
    cur = 0
    for i in range(size):
        cur += diff[i]
        if cur > capacity:
            return False
    return True


if __name__ == "__main__":
    assert car_pooling([[2, 1, 5], [3, 3, 7]], 4) is False
    assert car_pooling([[2, 1, 5], [3, 3, 7]], 5) is True
    assert car_pooling([[2, 1, 5], [3, 5, 7]], 5) is True
    assert car_pooling([], 1) is True
    assert car_pooling([[9, 0, 1]], 8) is False
    assert car_pooling([[3, 2, 4], [2, 4, 6]], 5) is True
    print("car_pooling: all tests passed")
