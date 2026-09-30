// 1046. 最后一块石头的重量
// 见 last_stone_weight.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <vector>

int lastStoneWeight(std::vector<int> stones) {
    std::priority_queue<int> maxHeap(stones.begin(), stones.end());
    while (maxHeap.size() > 1) {
        int first = maxHeap.top();
        maxHeap.pop();
        int second = maxHeap.top();
        maxHeap.pop();
        if (first != second) maxHeap.push(first - second);
    }
    return maxHeap.empty() ? 0 : maxHeap.top();
}

int main() {
    assert(lastStoneWeight({2, 7, 4, 1, 8, 1}) == 1);
    assert(lastStoneWeight({1}) == 1);
    assert(lastStoneWeight({2, 2}) == 0);
    assert(lastStoneWeight({1, 3}) == 2);
    assert(lastStoneWeight({3, 7, 2}) == 2);
    assert(lastStoneWeight({}) == 0);
    std::cout << "last_stone_weight: all tests passed\n";
    return 0;
}
