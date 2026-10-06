// 981. 基于时间的键值存储
// 见 time_map.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class TimeMap {
public:
    void set(const std::string &key, const std::string &value, int timestamp) {
        times_[key].push_back(timestamp);
        values_[key].push_back(value);
    }

    std::string get(const std::string &key, int timestamp) {
        auto it = times_.find(key);
        if (it == times_.end()) {
            return "";
        }
        const std::vector<int> &ts = it->second;
        int i = static_cast<int>(
                    std::upper_bound(ts.begin(), ts.end(), timestamp) - ts.begin()) -
                1;
        return i >= 0 ? values_[key][i] : "";
    }

private:
    std::unordered_map<std::string, std::vector<int>> times_;
    std::unordered_map<std::string, std::vector<std::string>> values_;
};

int main() {
    TimeMap tm;
    tm.set("foo", "bar", 1);
    assert(tm.get("foo", 1) == "bar");
    assert(tm.get("foo", 3) == "bar");   // 3 时刻最近的一版仍是 1 时刻的 bar
    tm.set("foo", "bar2", 4);
    assert(tm.get("foo", 4) == "bar2");
    assert(tm.get("foo", 5) == "bar2");
    assert(tm.get("foo", 3) == "bar");   // 落在两版之间，取更早那版
    assert(tm.get("foo", 0) == "");      // 早于首次 set
    assert(tm.get("missing", 10) == ""); // 从未出现过

    std::cout << "time_map: all tests passed\n";
    return 0;
}
