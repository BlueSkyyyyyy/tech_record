// 1146. 快照数组
// 见 snapshot_array.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

class SnapshotArray {
public:
    explicit SnapshotArray(int length)
        : history_(length, std::vector<std::pair<int, int>>{{-1, 0}}), snap_id_(0) {}

    void set(int index, int val) {
        std::vector<std::pair<int, int>> &row = history_[index];
        if (row.back().first == snap_id_) {
            row.back().second = val;
        } else {
            row.emplace_back(snap_id_, val);
        }
    }

    int snap() { return snap_id_++; }

    int get(int index, int snap_id) {
        const std::vector<std::pair<int, int>> &row = history_[index];
        int lo = 0, hi = static_cast<int>(row.size()) - 1, ans = 0;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (row[mid].first <= snap_id) {
                ans = row[mid].second;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return ans;
    }

private:
    std::vector<std::vector<std::pair<int, int>>> history_;
    int snap_id_;
};

int main() {
    SnapshotArray arr(3);
    arr.set(0, 5);
    assert(arr.snap() == 0);
    arr.set(0, 6);
    assert(arr.get(0, 0) == 5);
    assert(arr.get(1, 0) == 0);
    assert(arr.snap() == 1);
    assert(arr.get(0, 1) == 6);
    arr.set(2, 7);
    assert(arr.snap() == 2);
    assert(arr.get(2, 2) == 7);
    assert(arr.get(2, 1) == 0);
    assert(arr.get(0, 2) == 6);

    std::cout << "snapshot_array: all tests passed\n";
    return 0;
}
