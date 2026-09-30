// 215. 数组中的第 K 个最大元素
// 见 kth_largest_element_in_an_array.py 的题目与思路说明。
#include <cassert>
#include <functional>
#include <iostream>
#include <queue>
#include <vector>

int findKthLargest(const std::vector<int> &nums, int k) {
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap;
    for (int num : nums) {
        minHeap.push(num);
        if ((int)minHeap.size() > k) minHeap.pop();
    }
    return minHeap.top();
}

int main() {
    assert(findKthLargest({3, 2, 1, 5, 6, 4}, 2) == 5);
    assert(findKthLargest({3, 2, 3, 1, 2, 4, 5, 5, 6}, 4) == 4);
    assert(findKthLargest({1}, 1) == 1);
    assert(findKthLargest({-1, -2, -3}, 3) == -3);
    assert(findKthLargest({7, 7, 7}, 2) == 7);
    std::cout << "kth_largest_element_in_an_array: all tests passed\n";
    return 0;
}
