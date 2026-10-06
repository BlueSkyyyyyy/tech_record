// 380. O(1) 时间插入、删除和获取随机元素
// 见 insert_delete_getrandom.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class RandomizedSet {
public:
    RandomizedSet() { std::srand(12345); }

    bool insert(int val) {
        if (pos_.count(val)) {
            return false;
        }
        pos_[val] = static_cast<int>(vals_.size());
        vals_.push_back(val);
        return true;
    }

    bool remove(int val) {
        auto it = pos_.find(val);
        if (it == pos_.end()) {
            return false;
        }
        int idx = it->second;
        int last = vals_.back();
        vals_[idx] = last;
        pos_[last] = idx;
        vals_.pop_back();
        pos_.erase(val);
        return true;
    }

    int getRandom() {
        return vals_[std::rand() % vals_.size()];
    }

private:
    std::vector<int> vals_;
    std::unordered_map<int, int> pos_;
};

int main() {
    RandomizedSet s;
    assert(s.insert(1));
    assert(!s.remove(2));
    assert(s.insert(2));
    assert(s.remove(1));
    assert(!s.insert(2));
    assert(s.getRandom() == 2);

    for (int v = 0; v < 100; ++v) {
        s.insert(v);
    }
    assert(s.remove(37));
    assert(s.insert(37));

    std::unordered_set<int> seen;
    for (int i = 0; i < 2000; ++i) {
        int x = s.getRandom();
        assert((0 <= x && x <= 99) || x == 2);
        seen.insert(x);
    }
    assert(seen.size() >= 50);

    std::cout << "insert_delete_getrandom: all tests passed\n";
    return 0;
}
