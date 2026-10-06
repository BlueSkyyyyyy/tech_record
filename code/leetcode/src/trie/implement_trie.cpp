// 208. 实现 Trie（前缀树）
// 见 implement_trie.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <string>

class Trie {
  public:
    Trie() { children_.fill(nullptr); }
    ~Trie() {
        for (Trie *child : children_) delete child;
    }

    Trie(const Trie &) = delete;
    Trie &operator=(const Trie &) = delete;

    void insert(const std::string &word) {
        Trie *node = this;
        for (char ch : word) {
            int i = ch - 'a';
            if (!node->children_[i]) node->children_[i] = new Trie();
            node = node->children_[i];
        }
        node->isEnd_ = true;
    }

    bool search(const std::string &word) const {
        const Trie *node = find(word);
        return node != nullptr && node->isEnd_;
    }

    bool startsWith(const std::string &prefix) const {
        return find(prefix) != nullptr;
    }

  private:
    std::array<Trie *, 26> children_;
    bool isEnd_ = false;

    const Trie *find(const std::string &prefix) const {
        const Trie *node = this;
        for (char ch : prefix) {
            int i = ch - 'a';
            if (!node->children_[i]) return nullptr;
            node = node->children_[i];
        }
        return node;
    }
};

int main() {
    Trie trie;
    trie.insert("apple");
    assert(trie.search("apple"));
    assert(!trie.search("app"));
    assert(trie.startsWith("app"));

    trie.insert("app");
    assert(trie.search("app"));

    trie.insert("banana");
    assert(trie.startsWith("ban"));
    assert(!trie.search("ban"));
    assert(!trie.search("bandana"));
    std::cout << "implement_trie: all tests passed\n";
    return 0;
}
