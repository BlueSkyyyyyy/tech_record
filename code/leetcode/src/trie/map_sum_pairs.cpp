// 677. 键值映射
// 见 map_sum_pairs.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <string>

class MapSum {
  public:
    MapSum() { children_.fill(nullptr); }
    ~MapSum() {
        for (MapSum *child : children_) delete child;
    }

    MapSum(const MapSum &) = delete;
    MapSum &operator=(const MapSum &) = delete;

    void insert(const std::string &key, int val) {
        int delta = val - getValue(key);
        MapSum *node = this;
        for (char ch : key) {
            int i = ch - 'a';
            if (!node->children_[i]) node->children_[i] = new MapSum();
            node = node->children_[i];
            node->total_ += delta;
        }
        node->value_ = val;
    }

    int sum(const std::string &prefix) const {
        const MapSum *node = this;
        for (char ch : prefix) {
            int i = ch - 'a';
            if (!node->children_[i]) return 0;
            node = node->children_[i];
        }
        return node->total_;
    }

  private:
    std::array<MapSum *, 26> children_;
    int total_ = 0;
    int value_ = 0;

    int getValue(const std::string &key) const {
        const MapSum *node = this;
        for (char ch : key) {
            int i = ch - 'a';
            if (!node->children_[i]) return 0;
            node = node->children_[i];
        }
        return node->value_;
    }
};

int main() {
    MapSum m;
    m.insert("apple", 3);
    assert(m.sum("ap") == 3);
    m.insert("app", 2);
    assert(m.sum("ap") == 5);
    m.insert("apple", 2);  // 覆盖：apple 由 3 变成 2
    assert(m.sum("ap") == 4);
    assert(m.sum("app") == 4);  // app(2) + apple(2)，因 apple 也以 app 为前缀
    assert(m.sum("b") == 0);
    std::cout << "map_sum_pairs: all tests passed\n";
    return 0;
}
