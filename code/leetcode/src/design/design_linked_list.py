"""707. 设计链表（Design Linked List）

题目：设计一个链表，支持：
    get(index)：取第 index 个节点的值，越界返回 -1；
    addAtHead(val) / addAtTail(val)：头插 / 尾插；
    addAtIndex(index, val)：在第 index 个节点前插入；index == 长度则尾插，越界不插；
    deleteAtIndex(index)：删除第 index 个节点，越界不删。

思路（哨兵头尾 + 双向指针）：
    链表的痛点是边界：在头部插、在尾部插、删第一个、删最后一个，都要特判。办法是
    一开始就放两个不存真实数据的哨兵节点 `head` 和 `tail`，让它们互指，真实节点永远
    插在这两者之间。这样「头插」就是「在下标 0 处插」，「尾插」就是「在 length 处插」，
    删除也永远有前驱和后继可改，边界全部消失。

    `_node_at(index)` 负责把下标翻译成节点指针。它做一个小优化：从靠近的一端出发，
    index 在前半段就从 head 往右走，在后半段就从 tail 往左走，最多走 length/2 步。

复杂度：get / add / delete 时间 O(min(index, n-index))，其中 `_node_at` 是主要开销；
    空间 O(n)。
"""


class _Node:
    __slots__ = ("val", "prev", "next")

    def __init__(self, val=0):
        self.val = val
        self.prev = None
        self.next = None


class MyLinkedList:
    def __init__(self):
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0

    def _node_at(self, index):
        if index < 0 or index >= self.size:
            return None
        if index < self.size // 2:
            cur = self.head.next
            for _ in range(index):
                cur = cur.next
        else:
            cur = self.tail.prev
            for _ in range(self.size - 1 - index):
                cur = cur.prev
        return cur

    def get(self, index):
        node = self._node_at(index)
        return node.val if node else -1

    def addAtHead(self, val):
        self.addAtIndex(0, val)

    def addAtTail(self, val):
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index, val):
        if index < 0 or index > self.size:
            return
        nxt = self.tail if index == self.size else self._node_at(index)
        node = _Node(val)
        prev = nxt.prev
        node.prev = prev
        node.next = nxt
        prev.next = node
        nxt.prev = node
        self.size += 1

    def deleteAtIndex(self, index):
        node = self._node_at(index)
        if node is None:
            return
        node.prev.next = node.next
        node.next.prev = node.prev
        self.size -= 1


if __name__ == "__main__":
    ll = MyLinkedList()
    ll.addAtHead(1)
    ll.addAtTail(3)
    ll.addAtIndex(1, 2)          # 1 -> 2 -> 3
    assert ll.get(0) == 1
    assert ll.get(1) == 2
    assert ll.get(2) == 3
    assert ll.get(3) == -1
    assert ll.get(-1) == -1
    ll.deleteAtIndex(1)          # 1 -> 3
    assert ll.get(1) == 3
    assert ll.get(2) == -1
    ll.addAtTail(4)              # 1 -> 3 -> 4
    assert ll.get(2) == 4
    ll.deleteAtIndex(0)          # 3 -> 4
    assert ll.get(0) == 3
    ll.deleteAtIndex(5)          # 越界，不删
    assert ll.get(1) == 4
    ll.addAtIndex(2, 9)          # 尾插：3 -> 4 -> 9
    assert ll.get(2) == 9
    ll.addAtIndex(5, 9)          # 越界，不插
    assert ll.get(2) == 9
    print("design_linked_list: all tests passed")
