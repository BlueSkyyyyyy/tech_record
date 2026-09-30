// 973. 最接近原点的 K 个点
// 见 k_closest_points_to_origin.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

std::vector<std::vector<int>> kClosest(std::vector<std::vector<int>> &points, int k) {
    // 默认大顶堆，堆顶是当前保留点里距离平方最大的（最远），正好当淘汰线。
    std::priority_queue<std::pair<int, std::pair<int, int>>> maxHeap;
    for (const auto &point : points) {
        int x = point[0], y = point[1];
        maxHeap.push({x * x + y * y, {x, y}});
        if ((int)maxHeap.size() > k) maxHeap.pop();
    }

    std::vector<std::vector<int>> result;
    while (!maxHeap.empty()) {
        auto [dist, coord] = maxHeap.top();
        maxHeap.pop();
        result.push_back({coord.first, coord.second});
    }
    return result;
}

int main() {
    std::vector<std::vector<int>> points1 = {{1, 3}, {-2, 2}};
    std::vector<std::vector<int>> want1 = {{-2, 2}};
    std::vector<std::vector<int>> got1 = kClosest(points1, 1);
    std::sort(got1.begin(), got1.end());
    assert(got1 == want1);

    std::vector<std::vector<int>> points2 = {{3, 3}, {5, -1}, {-2, 4}};
    std::vector<std::vector<int>> want2 = {{-2, 4}, {3, 3}};
    std::vector<std::vector<int>> got2 = kClosest(points2, 2);
    std::sort(got2.begin(), got2.end());
    assert(got2 == want2);

    std::vector<std::vector<int>> points3 = {{0, 1}, {1, 0}};
    std::vector<std::vector<int>> want3 = {{0, 1}, {1, 0}};
    std::vector<std::vector<int>> got3 = kClosest(points3, 2);
    std::sort(got3.begin(), got3.end());
    assert(got3 == want3);

    std::vector<std::vector<int>> points4 = {{-5, 4}, {-6, -5}, {4, 6}};
    std::vector<std::vector<int>> want4 = {{-5, 4}};
    std::vector<std::vector<int>> got4 = kClosest(points4, 1);
    std::sort(got4.begin(), got4.end());
    assert(got4 == want4);
    std::cout << "k_closest_points_to_origin: all tests passed\n";
    return 0;
}
