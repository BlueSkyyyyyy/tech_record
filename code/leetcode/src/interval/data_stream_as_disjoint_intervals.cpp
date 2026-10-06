// 352. 将数据流变为多个不相交区间
// 见 data_stream_as_disjoint_intervals.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

class SummaryRanges {
public:
    void addNum(int value) {
        auto it = std::lower_bound(
            intervals.begin(), intervals.end(), value,
            [](const std::vector<int>& a, int v) { return a[0] < v; });

        if (it != intervals.begin() && (*(it - 1))[1] >= value) {
            return;
        }
        if (it != intervals.end() && (*it)[0] == value) {
            return;
        }

        bool leftAdj =
            (it != intervals.begin() && (*(it - 1))[1] == value - 1);
        bool rightAdj =
            (it != intervals.end() && (*it)[0] == value + 1);

        if (leftAdj && rightAdj) {
            (*(it - 1))[1] = (*it)[1];
            intervals.erase(it);
        } else if (leftAdj) {
            (*(it - 1))[1] = value;
        } else if (rightAdj) {
            (*it)[0] = value;
        } else {
            intervals.insert(it, {value, value});
        }
    }

    std::vector<std::vector<int>> getIntervals() { return intervals; }

private:
    std::vector<std::vector<int>> intervals;
};

int main() {
    SummaryRanges sr;
    sr.addNum(1);
    std::vector<std::vector<int>> w1{{1, 1}};
    assert(sr.getIntervals() == w1);
    sr.addNum(3);
    std::vector<std::vector<int>> w2{{1, 1}, {3, 3}};
    assert(sr.getIntervals() == w2);
    sr.addNum(7);
    std::vector<std::vector<int>> w3{{1, 1}, {3, 3}, {7, 7}};
    assert(sr.getIntervals() == w3);
    sr.addNum(2);
    std::vector<std::vector<int>> w4{{1, 3}, {7, 7}};
    assert(sr.getIntervals() == w4);
    sr.addNum(6);
    std::vector<std::vector<int>> w5{{1, 3}, {6, 7}};
    assert(sr.getIntervals() == w5);

    SummaryRanges sr2;
    sr2.addNum(5);
    sr2.addNum(5);
    std::vector<std::vector<int>> v1{{5, 5}};
    assert(sr2.getIntervals() == v1);
    sr2.addNum(4);
    sr2.addNum(3);
    std::vector<std::vector<int>> v2{{3, 5}};
    assert(sr2.getIntervals() == v2);

    std::cout << "data_stream_as_disjoint_intervals: all tests passed\n";
    return 0;
}
