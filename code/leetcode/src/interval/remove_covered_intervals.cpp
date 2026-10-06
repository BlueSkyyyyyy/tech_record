// 1288. 删除被覆盖区间
// 见 remove_covered_intervals.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int removeCoveredIntervals(std::vector<std::vector<int>>& intervals) {
    std::sort(intervals.begin(), intervals.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
                  if (a[0] != b[0]) {
                      return a[0] < b[0];
                  }
                  return a[1] > b[1];
              });
    int count = 0;
    int maxEnd = -1;
    for (const auto& iv : intervals) {
        if (iv[1] > maxEnd) {
            ++count;
            maxEnd = iv[1];
        }
    }
    return count;
}

int main() {
    std::vector<std::vector<int>> a{{1, 4}, {3, 6}, {2, 8}};
    std::vector<std::vector<int>> b{{1, 4}, {2, 3}};
    std::vector<std::vector<int>> c{{0, 10}, {5, 12}};
    std::vector<std::vector<int>> d{{3, 10}, {4, 10}, {5, 11}};
    std::vector<std::vector<int>> e{{1, 2}, {1, 4}, {3, 4}};

    assert(removeCoveredIntervals(a) == 2);
    assert(removeCoveredIntervals(b) == 1);
    assert(removeCoveredIntervals(c) == 2);
    assert(removeCoveredIntervals(d) == 2);
    assert(removeCoveredIntervals(e) == 1);

    std::cout << "remove_covered_intervals: all tests passed\n";
    return 0;
}
