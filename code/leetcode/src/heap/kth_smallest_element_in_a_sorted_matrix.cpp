// 378. 有序矩阵中第 K 小的元素
// 见 kth_smallest_element_in_a_sorted_matrix.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <tuple>
#include <vector>

int kthSmallest(const std::vector<std::vector<int>> &matrix, int k) {
    int n = matrix.size();
    using Item = std::tuple<int, int, int>;  // (值, 行, 列)
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> minHeap;
    for (int i = 0; i < n; ++i) minHeap.push({matrix[i][0], i, 0});

    for (int step = 0; step < k - 1; ++step) {
        auto [value, i, j] = minHeap.top();
        minHeap.pop();
        if (j + 1 < (int)matrix[i].size()) {
            minHeap.push({matrix[i][j + 1], i, j + 1});
        }
    }
    return std::get<0>(minHeap.top());
}

int main() {
    std::vector<std::vector<int>> matrix = {
        {1, 5, 9},
        {10, 11, 13},
        {12, 13, 15},
    };
    assert(kthSmallest(matrix, 8) == 13);
    assert(kthSmallest(matrix, 1) == 1);
    assert(kthSmallest(matrix, 9) == 15);
    assert(kthSmallest({{-5}}, 1) == -5);
    assert(kthSmallest({{1, 2}, {1, 3}}, 2) == 1);
    assert(kthSmallest({{1, 2}, {3, 4}}, 4) == 4);
    std::cout << "kth_smallest_element_in_a_sorted_matrix: all tests passed\n";
    return 0;
}
