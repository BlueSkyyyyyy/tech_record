"""1670. 设计前中后队列（Design Front Middle Back Queue）

题目：实现一个队列，支持 pushFront / pushMiddle / pushBack /
popFront / popMiddle / popBack。元素个数为偶数时，「中间」取靠前的那个。

思路（两个双端队列分担前后两半）：
    用一个双端队列 a 存前半段（含中间元素），另一个双端队列 b 存后半段，
    维持不变量：`len(a) == len(b)` 或 `len(a) == len(b) + 1`。
    这样 a 的队尾 `a[-1]` 永远就是「中间元素」，b 的队尾是「队尾元素」。

    - pushFront：进 a 队首，如果 a 比 b 多出 2 个，就把 a 队尾挪到 b 队首；
    - pushBack：进 b 队尾，如果 b 比 a 多了，就把 b 队首挪回 a 队尾；
    - pushMiddle：a 比 b 多时先把 a 队尾挪到 b 队首，再把新元素放进 a 队尾；
    - popFront / popMiddle：都从 a 取，取完若 a 变短了就补一个 b 队首过来；
    - popBack：从 b 取；b 空了说明只剩 a 里那一个，直接取 a；取完再平衡。

    所有「平衡」操作都只搬一个元素，因为它保证不变量最多被破坏 1。

复杂度：所有操作 O(1)，空间 O(n)。
"""

from collections import deque


class FrontMiddleBackQueue:
    def __init__(self):
        self.a = deque()
        self.b = deque()

    def pushFront(self, val):
        self.a.appendleft(val)
        if len(self.a) > len(self.b) + 1:
            self.b.appendleft(self.a.pop())

    def pushMiddle(self, val):
        if len(self.a) > len(self.b):
            self.b.appendleft(self.a.pop())
        self.a.append(val)

    def pushBack(self, val):
        self.b.append(val)
        if len(self.b) > len(self.a):
            self.a.append(self.b.popleft())

    def popFront(self):
        if not self.a:
            return -1
        val = self.a.popleft()
        if len(self.a) < len(self.b):
            self.a.append(self.b.popleft())
        return val

    def popMiddle(self):
        if not self.a:
            return -1
        val = self.a.pop()
        if len(self.a) < len(self.b):
            self.a.append(self.b.popleft())
        return val

    def popBack(self):
        if not self.b:
            return self.a.pop() if self.a else -1
        val = self.b.pop()
        if len(self.a) > len(self.b) + 1:
            self.b.appendleft(self.a.pop())
        return val


if __name__ == "__main__":
    q = FrontMiddleBackQueue()
    q.pushFront(1)
    q.pushBack(2)
    q.pushMiddle(3)
    q.pushMiddle(4)
    assert q.popFront() == 1
    assert q.popMiddle() == 3
    assert q.popMiddle() == 4
    assert q.popBack() == 2
    assert q.popFront() == -1

    q2 = FrontMiddleBackQueue()
    for v in range(1, 6):
        q2.pushBack(v)
    assert q2.popMiddle() == 3
    assert q2.popFront() == 1
    assert q2.popMiddle() == 4
    assert q2.popBack() == 5
    assert q2.popFront() == 2
    assert q2.popBack() == -1
    print("front_middle_back_queue: all tests passed")
