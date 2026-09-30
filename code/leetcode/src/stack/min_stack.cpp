// 155. 最小栈
// 见 min_stack.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <stack>

class MinStack {
  public:
    void push(int val) {
        stack_.push(val);
        if (minStack_.empty() || val <= minStack_.top()) minStack_.push(val);
    }

    void pop() {
        int val = stack_.top();
        stack_.pop();
        if (val == minStack_.top()) minStack_.pop();
    }

    int top() { return stack_.top(); }

    int getMin() { return minStack_.top(); }

  private:
    std::stack<int> stack_;
    std::stack<int> minStack_;
};

int main() {
    MinStack st;
    st.push(-2);
    st.push(0);
    st.push(-3);
    assert(st.getMin() == -3);
    st.pop();
    assert(st.top() == 0);
    assert(st.getMin() == -2);

    MinStack st2;
    st2.push(5);
    st2.push(5);
    st2.push(3);
    assert(st2.getMin() == 3);
    st2.pop();
    assert(st2.getMin() == 5);
    st2.pop();
    assert(st2.getMin() == 5);
    std::cout << "min_stack: all tests passed\n";
    return 0;
}
