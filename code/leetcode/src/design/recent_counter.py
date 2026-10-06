"""933. 最近的请求次数（Number of Recent Calls）

题目：实现 RecentCounter 类，只有一个方法 ping(t)：在时间 t 新增一次请求（t 单调递增），
返回过去 3000 毫秒内（即闭区间 [t - 3000, t]）发生的请求次数。

思路（普通队列 = 滑动窗口）：
    时间 t 单调递增，所以「过去 3000 毫秒」就是窗口的右端固定在最新请求、左端不超过
    t - 3000。把请求时间按先后放进一个队列：每次 ping 先入队，再把队首所有小于
    t - 3000 的时间弹出，剩下的元素个数正好落在窗口内。

    窗口右端随 t 单调右移，老请求一旦过期就再也不会回来，所以每个元素只进出队一次，
    摊还 O(1)。

复杂度：每次 ping 摊还时间 O(1)，空间 O(窗口内请求数)。
"""

from collections import deque


class RecentCounter:
    def __init__(self):
        self.q = deque()

    def ping(self, t):
        self.q.append(t)
        while self.q[0] < t - 3000:
            self.q.popleft()
        return len(self.q)


if __name__ == "__main__":
    counter = RecentCounter()
    assert counter.ping(1) == 1
    assert counter.ping(100) == 2
    assert counter.ping(3001) == 3
    assert counter.ping(3002) == 3          # 1 已过期
    assert counter.ping(7000) == 1          # 只剩 7000 自己
    assert counter.ping(7001) == 2
    print("recent_counter: all tests passed")
