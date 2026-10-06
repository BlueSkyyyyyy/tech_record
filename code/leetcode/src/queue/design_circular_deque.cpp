// 641. 设计循环双端队列
// 见 design_circular_deque.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

class MyCircularDeque {
public:
    explicit MyCircularDeque(int k) : cap_(k), data_(k, 0), head_(0), count_(0) {}

    bool insertFront(int value) {
        if (isFull()) {
            return false;
        }
        head_ = (head_ - 1 + cap_) % cap_;
        data_[head_] = value;
        ++count_;
        return true;
    }

    bool insertLast(int value) {
        if (isFull()) {
            return false;
        }
        data_[(head_ + count_) % cap_] = value;
        ++count_;
        return true;
    }

    bool deleteFront() {
        if (isEmpty()) {
            return false;
        }
        head_ = (head_ + 1) % cap_;
        --count_;
        return true;
    }

    bool deleteLast() {
        if (isEmpty()) {
            return false;
        }
        --count_;
        return true;
    }

    int getFront() const {
        if (isEmpty()) {
            return -1;
        }
        return data_[head_];
    }

    int getRear() const {
        if (isEmpty()) {
            return -1;
        }
        return data_[(head_ + count_ - 1) % cap_];
    }

    bool isEmpty() const { return count_ == 0; }
    bool isFull() const { return count_ == cap_; }

private:
    int cap_;
    std::vector<int> data_;
    int head_;
    int count_;
};

int main() {
    MyCircularDeque dq(3);
    assert(dq.insertLast(1) == true);
    assert(dq.insertLast(2) == true);
    assert(dq.insertFront(3) == true);
    assert(dq.insertFront(4) == false);
    assert(dq.getRear() == 2);
    assert(dq.isFull() == true);
    assert(dq.deleteLast() == true);
    assert(dq.insertFront(4) == true);
    assert(dq.getFront() == 4);

    MyCircularDeque dq2(1);
    assert(dq2.isEmpty() == true);
    assert(dq2.getFront() == -1);
    assert(dq2.insertFront(9) == true);
    assert(dq2.isFull() == true);
    assert(dq2.getRear() == 9);
    assert(dq2.deleteFront() == true);
    assert(dq2.isEmpty() == true);
    std::cout << "my_circular_deque: all tests passed\n";
    return 0;
}
