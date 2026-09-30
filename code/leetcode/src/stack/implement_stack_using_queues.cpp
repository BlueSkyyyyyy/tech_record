// 225. 用队列实现栈
// 见 implement_stack_using_queues.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>

class MyStack {
  public:
    void push(int x) {
        q_.push(x);
        int n = q_.size();
        for (int i = 0; i < n - 1; ++i) {
            q_.push(q_.front());
            q_.pop();
        }
    }

    int pop() {
        int val = q_.front();
        q_.pop();
        return val;
    }

    int top() { return q_.front(); }

    bool empty() { return q_.empty(); }

  private:
    std::queue<int> q_;
};

int main() {
    MyStack st;
    st.push(1);
    st.push(2);
    assert(st.top() == 2);
    assert(st.pop() == 2);
    assert(st.empty() == false);
    st.push(3);
    assert(st.pop() == 3);
    assert(st.pop() == 1);
    assert(st.empty() == true);
    std::cout << "implement_stack_using_queues: all tests passed\n";
    return 0;
}
