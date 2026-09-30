"""225. 用队列实现栈（Implement Stack using Queues）

题目：请你仅使用两个队列实现一个后入先出栈，支持栈的常规操作：
        - push(x)：将元素 x 压入栈顶；
        - pop()：移除并返回栈顶元素；
        - top()：返回栈顶元素；
        - empty()：如果栈为空，返回 True，否则返回 False。

思路：只用一个队列就够。push(x) 时先把 x 正常入队，
      再把队列中「除 x 以外的所有元素」依次出队、重新入队。
      这样刚压入的 x 就被转到了队头，而队头正是栈顶。

      为什么一轮旋转就能模拟栈：队列只能从尾进、从头出，天生是先进先出。
      但若每次新元素进来后都把它前面的所有元素搬到它后面，队头就永远是新元素，
      于是「后进」的元素先出，符合栈的语义。

      为什么不增加弹出成本：也可以选择 push 时 O(1)、pop 时旋转，两种做法等价。
      这里让 push 多做一点、pop/top 都直接取队头 O(1)，逻辑更对称。
      和上一题一样，目标是理解「用受限容器实现另一种语义」的思路，而非纠结常数。

复杂度：push 时间 O(n)，pop/top/empty 时间 O(1)，空间 O(n)。
"""

from collections import deque


class MyStack:
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return not self.q


if __name__ == "__main__":
    st = MyStack()
    st.push(1)
    st.push(2)
    assert st.top() == 2
    assert st.pop() == 2
    assert st.empty() is False
    st.push(3)
    assert st.pop() == 3
    assert st.pop() == 1
    assert st.empty() is True
    print("implement_stack_using_queues: all tests passed")
