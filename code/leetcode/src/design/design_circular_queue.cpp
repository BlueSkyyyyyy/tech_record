// 622. 设计循环队列
// 见 design_circular_queue.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <vector>

class MyCircularQueue {
public:
    explicit MyCircularQueue(int k)
        : data_(k, 0), capacity_(k), head_(0), count_(0) {}

    bool enQueue(int value) {
        if (count_ == capacity_) {
            return false;
        }
        data_[(head_ + count_) % capacity_] = value;
        ++count_;
        return true;
    }

    bool deQueue() {
        if (count_ == 0) {
            return false;
        }
        head_ = (head_ + 1) % capacity_;
        --count_;
        return true;
    }

    int Front() const {
        return count_ == 0 ? -1 : data_[head_];
    }

    int Rear() const {
        if (count_ == 0) {
            return -1;
        }
        return data_[(head_ + count_ - 1) % capacity_];
    }

    bool isEmpty() const { return count_ == 0; }

    bool isFull() const { return count_ == capacity_; }

private:
    std::vector<int> data_;
    int capacity_;
    int head_;
    int count_;
};

int main() {
    MyCircularQueue q(3);
    assert(q.enQueue(1));
    assert(q.enQueue(2));
    assert(q.enQueue(3));
    assert(!q.enQueue(4));             // 队满
    assert(q.Rear() == 3);
    assert(q.isFull());
    assert(q.deQueue());
    assert(q.enQueue(4));              // 绕回空位
    assert(q.Rear() == 4);
    assert(q.Front() == 2);
    assert(q.deQueue());
    assert(q.deQueue());
    assert(q.deQueue());
    assert(!q.deQueue());              // 队空
    assert(q.isEmpty());
    assert(q.Front() == -1);
    assert(q.Rear() == -1);

    std::cout << "design_circular_queue: all tests passed\n";
    return 0;
}
