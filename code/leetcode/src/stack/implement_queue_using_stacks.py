"""232. 用栈实现队列（Implement Queue using Stacks）

题目：请你仅使用两个栈实现先入先出队列，支持队列的常规操作：
        - push(x)：将元素 x 推到队列的末尾；
        - pop()：从队列开头移除并返回元素；
        - peek()：返回队列开头的元素；
        - empty()：如果队列为空，返回 True，否则返回 False。

思路：用两个栈分工——in_stack 只负责接收 push，out_stack 只负责弹出。
        push 时直接压入 in_stack；
        需要 pop/peek 时，如果 out_stack 为空，就把 in_stack 里的元素全部倒进 out_stack，
        然后从 out_stack 取栈顶。倒一次之后，元素的相对顺序就被反转成了队列顺序。

      为什么这样能得到队列顺序：栈把顺序反转一次，两个栈就反转两次、变回原序。
      把 in_stack 的元素逐个弹出并压入 out_stack，最早进来的元素会跑到 out_stack 栈顶，
      正好是要先出的队头。

      为什么「只在 out_stack 为空时才倒」：如果每次都倒，会把已经排好序的元素重新搅乱，
      也不必要。out_stack 里还有元素时，它的栈顶仍是更早入队的，直接取即可。

      为什么整体是摊还 O(1)：每个元素最多从 in_stack 搬到 out_stack 一次，
      n 次操作总搬运次数不超过 n，均摊到每次操作就是常数。

复杂度：push 时间 O(1)；pop/peek 摊还 O(1)；空间 O(n)。
"""


class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        self.peek()
        return self.out_stack.pop()

    def peek(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1]

    def empty(self):
        return not self.in_stack and not self.out_stack


if __name__ == "__main__":
    q = MyQueue()
    q.push(1)
    q.push(2)
    assert q.peek() == 1
    assert q.pop() == 1
    assert q.empty() is False
    q.push(3)
    assert q.pop() == 2
    assert q.pop() == 3
    assert q.empty() is True
    print("implement_queue_using_stacks: all tests passed")
