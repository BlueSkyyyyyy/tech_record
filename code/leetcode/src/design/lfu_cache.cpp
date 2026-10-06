// 460. LFU 缓存
// 见 lfu_cache.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <list>
#include <unordered_map>
#include <utility>

class LFUCache {
public:
    explicit LFUCache(int capacity) : capacity_(capacity), min_freq_(0) {}

    int get(int key) {
        auto it = info_.find(key);
        if (it == info_.end()) {
            return -1;
        }
        int value = it->second.first;
        touch(key);
        return value;
    }

    void put(int key, int value) {
        if (capacity_ <= 0) {
            return;
        }
        auto it = info_.find(key);
        if (it != info_.end()) {
            it->second.first = value;
            touch(key);
            return;
        }
        if (static_cast<int>(info_.size()) >= capacity_) {
            std::list<int> &bucket = buckets_[min_freq_];
            int evict = bucket.back();          // 该频次里最久未使用
            bucket.pop_back();
            pos_.erase(evict);
            info_.erase(evict);
            if (bucket.empty()) {
                buckets_.erase(min_freq_);
            }
        }
        info_[key] = {value, 1};
        buckets_[1].push_front(key);
        pos_[key] = buckets_[1].begin();
        min_freq_ = 1;
    }

private:
    void touch(int key) {
        int freq = info_[key].second;
        auto &bucket = buckets_[freq];
        bucket.erase(pos_[key]);
        pos_.erase(key);
        if (bucket.empty()) {
            buckets_.erase(freq);
            if (min_freq_ == freq) {
                ++min_freq_;
            }
        }
        int new_freq = freq + 1;
        info_[key].second = new_freq;
        buckets_[new_freq].push_front(key);
        pos_[key] = buckets_[new_freq].begin();
    }

    int capacity_;
    int min_freq_;
    std::unordered_map<int, std::pair<int, int>> info_;       // key -> (value, freq)
    std::unordered_map<int, std::list<int>> buckets_;         // freq -> keys
    std::unordered_map<int, std::list<int>::iterator> pos_;   // key -> 在桶里的位置
};

int main() {
    LFUCache cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    assert(cache.get(1) == 1);       // key1 频次升到 2
    cache.put(3, 3);                 // 淘汰频次最低的 key2（频次 1）
    assert(cache.get(2) == -1);
    assert(cache.get(3) == 3);       // key3 频次升到 2
    cache.put(4, 4);                 // key1 与 key3 频次都是 2，淘汰更久没用的 key1
    assert(cache.get(1) == -1);
    assert(cache.get(3) == 3);
    assert(cache.get(4) == 4);

    LFUCache zero(0);
    zero.put(1, 1);
    assert(zero.get(1) == -1);

    std::cout << "lfu_cache: all tests passed\n";
    return 0;
}
