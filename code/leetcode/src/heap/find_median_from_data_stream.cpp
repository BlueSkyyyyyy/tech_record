// 295. 数据流的中位数
// 见 find_median_from_data_stream.py 的题目与思路说明。
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <queue>
#include <vector>

class MedianFinder {
public:
    void addNum(int num) {
        left_.push(num);
        right_.push(left_.top());
        left_.pop();
        if (right_.size() > left_.size()) {
            left_.push(right_.top());
            right_.pop();
        }
    }

    double findMedian() const {
        if (left_.size() > right_.size()) return left_.top();
        return (left_.top() + right_.top()) / 2.0;
    }

private:
    std::priority_queue<int> left_;                                  // 大顶堆，较小的一半
    std::priority_queue<int, std::vector<int>, std::greater<int>> right_;  // 小顶堆，较大的一半
};

int main() {
    MedianFinder finder;
    finder.addNum(1);
    finder.addNum(2);
    assert(std::fabs(finder.findMedian() - 1.5) < 1e-9);
    finder.addNum(3);
    assert(std::fabs(finder.findMedian() - 2.0) < 1e-9);

    MedianFinder finder2;
    for (int num : {6, 10, 2, 6, 5, 0, 6, 3, 1, 0, 0}) finder2.addNum(num);
    assert(std::fabs(finder2.findMedian() - 3.0) < 1e-9);

    MedianFinder finder3;
    finder3.addNum(-1);
    assert(std::fabs(finder3.findMedian() - (-1.0)) < 1e-9);
    finder3.addNum(-2);
    assert(std::fabs(finder3.findMedian() - (-1.5)) < 1e-9);
    std::cout << "find_median_from_data_stream: all tests passed\n";
    return 0;
}
