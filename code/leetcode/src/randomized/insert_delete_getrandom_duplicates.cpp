// 381. O(1) 时间插入、删除和获取随机元素 - 允许重复
// 见 insert_delete_getrandom_duplicates.py 的题目与思路说明。
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class RandomizedCollection {
public:
    RandomizedCollection() {}

    bool insert(int val) {
        nums_.push_back(val);
        pos_[val].insert(static_cast<int>(nums_.size()) - 1);
        return pos_[val].size() == 1;
    }

    bool remove(int val) {
        auto it = pos_.find(val);
        if (it == pos_.end()) {
            return false;
        }
        auto& idxs = it->second;
        int i = *idxs.begin();
        idxs.erase(idxs.begin());
        int lastIdx = static_cast<int>(nums_.size()) - 1;
        int last = nums_.back();
        if (i != lastIdx) {
            pos_[last].erase(lastIdx);
            pos_[last].insert(i);
        }
        nums_[i] = last;
        nums_.pop_back();
        if (idxs.empty()) {
            pos_.erase(it);
        }
        return true;
    }

    int getRandom() {
        return nums_[std::rand() % nums_.size()];
    }

private:
    std::vector<int> nums_;
    std::unordered_map<int, std::unordered_set<int>> pos_;
};

int main() {
    std::srand(12345);
    RandomizedCollection c;
    assert(c.insert(1));
    assert(!c.insert(1));
    assert(c.insert(2));
    assert(c.remove(2));
    assert(!c.remove(2));
    for (int t = 0; t < 200; ++t) {
        assert(c.getRandom() == 1);
    }

    RandomizedCollection c2;
    int vals[] = {4, 3, 4, 2, 4};
    for (int v : vals) {
        c2.insert(v);
    }
    std::unordered_set<int> seen;
    for (int t = 0; t < 400; ++t) {
        seen.insert(c2.getRandom());
    }
    assert(seen.size() == 3);
    assert(c2.remove(4));
    bool a = c2.remove(4);
    bool b = c2.remove(4);
    assert(a && b);
    assert(!c2.remove(4));

    std::unordered_set<int> seen2;
    for (int t = 0; t < 200; ++t) {
        seen2.insert(c2.getRandom());
    }
    assert(seen2.size() == 2);

    std::cout << "insert_delete_getrandom_duplicates: all tests passed\n";
    return 0;
}
