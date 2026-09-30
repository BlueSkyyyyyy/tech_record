// 703. 数据流中的第 K 大元素
// 见 kth_largest_element_in_a_stream.py 的题目与思路说明。
#include <cassert>
#include <functional>
#include <iostream>
#include <queue>
#include <vector>

class KthLargest {
public:
    KthLargest(int k, const std::vector<int> &nums) : k_(k) {
        for (int num : nums) add(num);
    }

    int add(int val) {
        minHeap_.push(val);
        if ((int)minHeap_.size() > k_) minHeap_.pop();
        return minHeap_.top();
    }

private:
    int k_;
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap_;
};

int main() {
    KthLargest kth(3, {4, 5, 8, 2});
    assert(kth.add(3) == 4);
    assert(kth.add(5) == 5);
    assert(kth.add(10) == 5);
    assert(kth.add(9) == 8);
    assert(kth.add(4) == 8);

    KthLargest kth2(1, {});
    assert(kth2.add(-3) == -3);
    assert(kth2.add(-2) == -2);
    assert(kth2.add(-4) == -2);
    assert(kth2.add(0) == 0);
    assert(kth2.add(4) == 4);

    KthLargest kth3(2, {0});
    assert(kth3.add(-1) == -1);
    assert(kth3.add(3) == 0);
    assert(kth3.add(5) == 3);
    std::cout << "kth_largest_element_in_a_stream: all tests passed\n";
    return 0;
}
