// 1670. 设计前中后队列
// 见 design_front_middle_back_queue.py 的题目与思路说明。
#include <cassert>
#include <deque>
#include <iostream>

class FrontMiddleBackQueue {
public:
    void pushFront(int val) {
        a_.push_front(val);
        if (static_cast<int>(a_.size()) > static_cast<int>(b_.size()) + 1) {
            b_.push_front(a_.back());
            a_.pop_back();
        }
    }

    void pushMiddle(int val) {
        if (a_.size() > b_.size()) {
            b_.push_front(a_.back());
            a_.pop_back();
        }
        a_.push_back(val);
    }

    void pushBack(int val) {
        b_.push_back(val);
        if (b_.size() > a_.size()) {
            a_.push_back(b_.front());
            b_.pop_front();
        }
    }

    int popFront() {
        if (a_.empty()) {
            return -1;
        }
        int val = a_.front();
        a_.pop_front();
        if (a_.size() < b_.size()) {
            a_.push_back(b_.front());
            b_.pop_front();
        }
        return val;
    }

    int popMiddle() {
        if (a_.empty()) {
            return -1;
        }
        int val = a_.back();
        a_.pop_back();
        if (a_.size() < b_.size()) {
            a_.push_back(b_.front());
            b_.pop_front();
        }
        return val;
    }

    int popBack() {
        if (b_.empty()) {
            if (a_.empty()) {
                return -1;
            }
            int val = a_.back();
            a_.pop_back();
            return val;
        }
        int val = b_.back();
        b_.pop_back();
        if (static_cast<int>(a_.size()) > static_cast<int>(b_.size()) + 1) {
            b_.push_front(a_.back());
            a_.pop_back();
        }
        return val;
    }

private:
    std::deque<int> a_;
    std::deque<int> b_;
};

int main() {
    FrontMiddleBackQueue q;
    q.pushFront(1);
    q.pushBack(2);
    q.pushMiddle(3);
    q.pushMiddle(4);
    assert(q.popFront() == 1);
    assert(q.popMiddle() == 3);
    assert(q.popMiddle() == 4);
    assert(q.popBack() == 2);
    assert(q.popFront() == -1);

    FrontMiddleBackQueue q2;
    for (int v = 1; v <= 5; ++v) {
        q2.pushBack(v);
    }
    assert(q2.popMiddle() == 3);
    assert(q2.popFront() == 1);
    assert(q2.popMiddle() == 4);
    assert(q2.popBack() == 5);
    assert(q2.popFront() == 2);
    assert(q2.popBack() == -1);
    std::cout << "front_middle_back_queue: all tests passed\n";
    return 0;
}
