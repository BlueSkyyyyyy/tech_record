// 1032. 字符流
// 见 stream_checker.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

struct TrieNode {
    std::array<TrieNode *, 26> child;
    bool isEnd = false;
    TrieNode() { child.fill(nullptr); }
};

class StreamChecker {
  public:
    explicit StreamChecker(const std::vector<std::string> &words) {
        root_ = new TrieNode();
        for (const std::string &word : words) {
            TrieNode *node = root_;
            for (auto it = word.rbegin(); it != word.rend(); ++it) {
                int i = *it - 'a';
                if (!node->child[i]) node->child[i] = new TrieNode();
                node = node->child[i];
            }
            node->isEnd = true;
        }
    }

    bool query(char letter) {
        stream_ += letter;
        TrieNode *node = root_;
        for (auto it = stream_.rbegin(); it != stream_.rend(); ++it) {
            int i = *it - 'a';
            if (!node->child[i]) return false;
            node = node->child[i];
            if (node->isEnd) return true;
        }
        return false;
    }

  private:
    TrieNode *root_ = nullptr;
    std::string stream_;
};

int main() {
    StreamChecker checker({"cd", "f", "kl"});
    std::string letters = "abcdefghijkl";
    std::vector<bool> want = {false, false, false, true, false, true, false,
                              false, false, false, false, true};
    for (size_t i = 0; i < letters.size(); ++i)
        assert(checker.query(letters[i]) == want[i]);

    StreamChecker single({"a"});
    assert(!single.query('b'));
    assert(single.query('a'));
    assert(single.query('a'));
    std::cout << "stream_checker: all tests passed\n";
    return 0;
}
