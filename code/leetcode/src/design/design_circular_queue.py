"""622. 设计循环队列（Design Circular Queue）

题目：设计一个循环队列，容量固定为 k，支持：
    enQueue(value)：入队，成功返回 True，队满返回 False；
    deQueue()：出队，成功返回 True，队空返回 False；
    Front() / Rear()：取队首 / 队尾元素，队空返回 -1；
    isEmpty() / isFull()：判空 / 判满。

思路（定长数组 + 队首下标 + 元素个数）：
    普通数组做队列，出队后前面会空出一段，只能整体搬移。循环队列让下标绕回来：
    新元素放在下标 `(head + count) % k` 处，出队只需 `head = (head + 1) % k`。
    存一个 `count` 而不是单独维护 tail，是因为「队空」和「队满」在环形数组里都会让
    head == tail，靠 count 才能区分二者：count == 0 是空，count == k 是满。
    队尾元素下标则是 `(head + count - 1) % k`。

复杂度：所有操作时间 O(1)，空间 O(k)。
"""


class MyCircularQueue:
    def __init__(self, k):
        self.data = [0] * k
        self.capacity = k
        self.head = 0
        self.count = 0

    def enQueue(self, value):
        if self.count == self.capacity:
            return False
        self.data[(self.head + self.count) % self.capacity] = value
        self.count += 1
        return True

    def deQueue(self):
        if self.count == 0:
            return False
        self.head = (self.head + 1) % self.capacity
        self.count -= 1
        return True

    def Front(self):
        return -1 if self.count == 0 else self.data[self.head]

    def Rear(self):
        if self.count == 0:
            return -1
        return self.data[(self.head + self.count - 1) % self.capacity]

    def isEmpty(self):
        return self.count == 0

    def isFull(self):
        return self.count == self.capacity


if __name__ == "__main__":
    q = MyCircularQueue(3)
    assert q.enQueue(1) is True
    assert q.enQueue(2) is True
    assert q.enQueue(3) is True
    assert q.enQueue(4) is False       # 队满
    assert q.Rear() == 3
    assert q.isFull() is True
    assert q.deQueue() is True
    assert q.enQueue(4) is True        # 绕回空位
    assert q.Rear() == 4
    assert q.Front() == 2
    assert q.deQueue() is True
    assert q.deQueue() is True
    assert q.deQueue() is True
    assert q.deQueue() is False        # 队空
    assert q.isEmpty() is True
    assert q.Front() == -1
    assert q.Rear() == -1
    print("design_circular_queue: all tests passed")
