// 232. 用栈实现队列
// 见 implement_queue_using_stacks.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <stack>

class MyQueue {
  public:
    void push(int x) { inStack_.push(x); }

    int pop() {
        peek();
        int val = outStack_.top();
        outStack_.pop();
        return val;
    }

    int peek() {
        if (outStack_.empty()) {
            while (!inStack_.empty()) {
                outStack_.push(inStack_.top());
                inStack_.pop();
            }
        }
        return outStack_.top();
    }

    bool empty() { return inStack_.empty() && outStack_.empty(); }

  private:
    std::stack<int> inStack_;
    std::stack<int> outStack_;
};

int main() {
    MyQueue q;
    q.push(1);
    q.push(2);
    assert(q.peek() == 1);
    assert(q.pop() == 1);
    assert(q.empty() == false);
    q.push(3);
    assert(q.pop() == 2);
    assert(q.pop() == 3);
    assert(q.empty() == true);
    std::cout << "implement_queue_using_stacks: all tests passed\n";
    return 0;
}
