"""155. 最小栈（Min Stack）

题目：设计一个支持 push、pop、top 操作，并能在常数时间内检索到最小元素的栈。
      需要实现 MinStack 类：
        - push(val)：将元素 val 压入栈；
        - pop()：删除栈顶元素；
        - top()：获取栈顶元素；
        - get_min()：检索栈中的最小元素。

思路：用一个主栈存所有元素，再开一个「最小栈」，让它与主栈同步：
        主栈每压入一个 val，若 val 不大于最小栈栈顶，就把 val 也压入最小栈；
        主栈每弹出一个 val，若它恰好等于最小栈栈顶，就同步弹出最小栈栈顶。
      这样最小栈的栈顶永远等于当前主栈里的最小值，get_min 直接取栈顶即可。

      为什么最小栈要「压入不大于栈顶的值」而不是「只记录更小的值」：
      因为要处理重复的最小值。若最小栈只在严格更小时才压入，那么当栈里有两个相同最小值、
      弹掉一个后，最小栈顶就提前变了，会得到错误结果。加上等号，弹出时也能一一对应地弹。

      为什么 get_min 是 O(1)：最小值始终被缓存在最小栈的栈顶，不需要现场遍历主栈。

复杂度：各操作时间 O(1)，空间 O(n)（最坏情况下最小栈与主栈等长）。
"""


class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]


if __name__ == "__main__":
    st = MinStack()
    st.push(-2)
    st.push(0)
    st.push(-3)
    assert st.get_min() == -3
    st.pop()
    assert st.top() == 0
    assert st.get_min() == -2

    st2 = MinStack()
    st2.push(5)
    st2.push(5)
    st2.push(3)
    assert st2.get_min() == 3
    st2.pop()
    assert st2.get_min() == 5
    st2.pop()
    assert st2.get_min() == 5
    print("min_stack: all tests passed")
