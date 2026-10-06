// 211. 添加与搜索单词 - 数据结构设计
// 见 add_and_search_words.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <string>

class WordDictionary {
  public:
    WordDictionary() { children_.fill(nullptr); }
    ~WordDictionary() {
        for (WordDictionary *child : children_) delete child;
    }

    WordDictionary(const WordDictionary &) = delete;
    WordDictionary &operator=(const WordDictionary &) = delete;

    void addWord(const std::string &word) {
        WordDictionary *node = this;
        for (char ch : word) {
            int i = ch - 'a';
            if (!node->children_[i]) node->children_[i] = new WordDictionary();
            node = node->children_[i];
        }
        node->isEnd_ = true;
    }

    bool search(const std::string &word) const { return dfs(this, word, 0); }

  private:
    std::array<WordDictionary *, 26> children_;
    bool isEnd_ = false;

    bool dfs(const WordDictionary *node, const std::string &word, int i) const {
        if (i == static_cast<int>(word.size())) return node->isEnd_;
        char ch = word[i];
        if (ch == '.') {
            for (const WordDictionary *child : node->children_) {
                if (child && dfs(child, word, i + 1)) return true;
            }
            return false;
        }
        const WordDictionary *child = node->children_[ch - 'a'];
        if (!child) return false;
        return dfs(child, word, i + 1);
    }
};

int main() {
    WordDictionary wd;
    wd.addWord("bad");
    wd.addWord("dad");
    wd.addWord("mad");
    assert(!wd.search("pad"));
    assert(wd.search("bad"));
    assert(wd.search(".ad"));
    assert(wd.search("b.."));

    wd.addWord("a");
    assert(wd.search("."));
    assert(!wd.search(".."));
    std::cout << "add_and_search_words: all tests passed\n";
    return 0;
}
