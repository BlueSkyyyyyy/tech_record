// 146. LRU 缓存
// 见 lru_cache.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <unordered_map>

struct Node {
    int key;
    int value;
    Node *prev;
    Node *next;
    Node(int k, int v) : key(k), value(v), prev(nullptr), next(nullptr) {}
};

class LRUCache {
public:
    explicit LRUCache(int capacity) : capacity_(capacity) {
        head_ = new Node(0, 0);
        tail_ = new Node(0, 0);
        head_->next = tail_;
        tail_->prev = head_;
    }

    int get(int key) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return -1;
        }
        Node *node = it->second;
        remove(node);
        addFront(node);
        return node->value;
    }

    void put(int key, int value) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            Node *node = it->second;
            node->value = value;
            remove(node);
            addFront(node);
            return;
        }
        if (static_cast<int>(cache_.size()) == capacity_) {
            Node *lru = tail_->prev;
            remove(lru);
            cache_.erase(lru->key);
            delete lru;
        }
        Node *node = new Node(key, value);
        cache_[key] = node;
        addFront(node);
    }

private:
    void remove(Node *node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void addFront(Node *node) {
        node->next = head_->next;
        node->prev = head_;
        head_->next->prev = node;
        head_->next = node;
    }

    int capacity_;
    std::unordered_map<int, Node *> cache_;
    Node *head_;
    Node *tail_;
};

int main() {
    LRUCache cache(2);
    cache.put(1, 1);
    cache.put(2, 2);
    assert(cache.get(1) == 1);       // 1 变为最近使用，2 变最旧
    cache.put(3, 3);                 // 淘汰 2
    assert(cache.get(2) == -1);
    assert(cache.get(3) == 3);
    cache.put(4, 4);                 // 淘汰 1
    assert(cache.get(1) == -1);
    assert(cache.get(3) == 3);
    assert(cache.get(4) == 4);
    cache.put(3, 30);                // 更新 3 的值
    assert(cache.get(3) == 30);
    cache.put(5, 5);                 // 淘汰最旧的 4
    assert(cache.get(4) == -1);
    assert(cache.get(3) == 30);
    assert(cache.get(5) == 5);

    LRUCache small(1);
    small.put(1, 1);
    small.put(2, 2);
    assert(small.get(1) == -1);
    assert(small.get(2) == 2);

    std::cout << "lru_cache: all tests passed\n";
    return 0;
}
