// 435. 无重叠区间
// 见 non_overlapping_intervals.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

int eraseOverlapIntervals(std::vector<std::vector<int>> intervals) {
    if (intervals.empty()) return 0;
    std::sort(intervals.begin(), intervals.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) {
                  return a[1] < b[1];
              });
    int end = intervals[0][1];
    int removed = 0;
    for (int i = 1; i < static_cast<int>(intervals.size()); ++i) {
        if (intervals[i][0] < end) {
            ++removed;
        } else {
            end = intervals[i][1];
        }
    }
    return removed;
}

int main() {
    std::vector<std::vector<int>> a = {{1, 2}, {2, 3}, {3, 4}, {1, 3}};
    assert(eraseOverlapIntervals(a) == 1);
    std::vector<std::vector<int>> b = {{1, 2}, {1, 2}, {1, 2}};
    assert(eraseOverlapIntervals(b) == 2);
    std::vector<std::vector<int>> c = {{1, 2}, {2, 3}};
    assert(eraseOverlapIntervals(c) == 0);
    std::vector<std::vector<int>> d;
    assert(eraseOverlapIntervals(d) == 0);
    std::vector<std::vector<int>> e = {{1, 100}, {11, 22}, {1, 11}, {2, 12}};
    assert(eraseOverlapIntervals(e) == 2);
    std::cout << "non_overlapping_intervals: all tests passed\n";
    return 0;
}
