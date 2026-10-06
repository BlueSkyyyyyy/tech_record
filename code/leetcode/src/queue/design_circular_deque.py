"""641. 设计循环双端队列（Design Circular Deque）

题目：实现一个固定容量 k 的循环双端队列，支持 insertFront / insertLast /
deleteFront / deleteLast / getFront / getRear / isEmpty / isFull。

思路（环形数组 + head + count）：
    用一个长度 k 的数组当环，`head` 指向队首元素的下标，`count` 记录当前元素个数。
    - 空：count == 0；满：count == k。用 count 而不是「head == tail」来区分空满，
      避免多留一个空位。
    - 队尾元素的下标是 `(head + count - 1) % k`。
    - 头部插入：head 往前挪一格（`(head - 1 + k) % k`）再写入；
      尾部插入：直接在 `(head + count) % k` 写入；
      头部删除：head 往后挪一格；尾部删除：count 减一即可。
    所有下标运算都要 `% k`，这就是「循环」的全部含义。

复杂度：所有操作 O(1)，空间 O(k)。
"""


class MyCircularDeque:
    def __init__(self, k):
        self.cap = k
        self.data = [0] * k
        self.head = 0
        self.count = 0

    def insertFront(self, value):
        if self.isFull():
            return False
        self.head = (self.head - 1) % self.cap
        self.data[self.head] = value
        self.count += 1
        return True

    def insertLast(self, value):
        if self.isFull():
            return False
        self.data[(self.head + self.count) % self.cap] = value
        self.count += 1
        return True

    def deleteFront(self):
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.cap
        self.count -= 1
        return True

    def deleteLast(self):
        if self.isEmpty():
            return False
        self.count -= 1
        return True

    def getFront(self):
        if self.isEmpty():
            return -1
        return self.data[self.head]

    def getRear(self):
        if self.isEmpty():
            return -1
        return self.data[(self.head + self.count - 1) % self.cap]

    def isEmpty(self):
        return self.count == 0

    def isFull(self):
        return self.count == self.cap


if __name__ == "__main__":
    dq = MyCircularDeque(3)
    assert dq.insertLast(1) is True
    assert dq.insertLast(2) is True
    assert dq.insertFront(3) is True
    assert dq.insertFront(4) is False
    assert dq.getRear() == 2
    assert dq.isFull() is True
    assert dq.deleteLast() is True
    assert dq.insertFront(4) is True
    assert dq.getFront() == 4

    dq2 = MyCircularDeque(1)
    assert dq2.isEmpty() is True
    assert dq2.getFront() == -1
    assert dq2.insertFront(9) is True
    assert dq2.isFull() is True
    assert dq2.getRear() == 9
    assert dq2.deleteFront() is True
    assert dq2.isEmpty() is True
    print("my_circular_deque: all tests passed")
