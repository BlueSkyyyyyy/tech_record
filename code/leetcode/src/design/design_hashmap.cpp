// 706. 设计哈希映射
// 见 design_hashmap.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>

class MyHashMap {
public:
    MyHashMap() : buckets_(kSize), size_(kSize) {}

    void put(int key, int value) {
        auto &bucket = buckets_[key % size_];
        for (auto &pair : bucket) {
            if (pair.first == key) {
                pair.second = value;
                return;
            }
        }
        bucket.emplace_back(key, value);
    }

    int get(int key) {
        for (const auto &pair : buckets_[key % size_]) {
            if (pair.first == key) {
                return pair.second;
            }
        }
        return -1;
    }

    void remove(int key) {
        auto &bucket = buckets_[key % size_];
        for (size_t i = 0; i < bucket.size(); ++i) {
            if (bucket[i].first == key) {
                bucket.erase(bucket.begin() + i);
                return;
            }
        }
    }

private:
    static const int kSize = 769;
    std::vector<std::vector<std::pair<int, int>>> buckets_;
    int size_;
};

int main() {
    MyHashMap m;
    m.put(1, 1);
    m.put(2, 2);
    assert(m.get(1) == 1);
    assert(m.get(3) == -1);
    m.put(2, 1);
    assert(m.get(2) == 1);
    m.remove(2);
    assert(m.get(2) == -1);
    m.remove(2);
    assert(m.get(2) == -1);
    m.put(769, 100);          // 与 key=0 同桶，检验链地址法
    m.put(0, 7);
    assert(m.get(769) == 100);
    assert(m.get(0) == 7);
    m.remove(769);
    assert(m.get(769) == -1);
    assert(m.get(0) == 7);

    std::cout << "design_hashmap: all tests passed\n";
    return 0;
}
