// 745. 前缀和后缀搜索
// 见 word_filter.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

struct TrieNode {
    std::array<TrieNode *, 27> child;  // 0..25 为 a..z，26 存分隔符 '{'
    int best = -1;
    TrieNode() { child.fill(nullptr); }
};

class WordFilter {
  public:
    explicit WordFilter(const std::vector<std::string> &words) {
        root_ = new TrieNode();
        for (int index = 0; index < static_cast<int>(words.size()); ++index) {
            const std::string &word = words[index];
            for (size_t start = 0; start <= word.size(); ++start) {
                std::string key = word.substr(start) + "{" + word;
                TrieNode *node = root_;
                for (char ch : key) {
                    int idx = indexOf(ch);
                    if (!node->child[idx]) node->child[idx] = new TrieNode();
                    node = node->child[idx];
                    node->best = index;
                }
            }
        }
    }

    int f(const std::string &prefix, const std::string &suffix) const {
        TrieNode *node = root_;
        for (char ch : suffix + "{" + prefix) {
            if (!node) return -1;
            node = node->child[indexOf(ch)];
        }
        return node ? node->best : -1;
    }

  private:
    TrieNode *root_ = nullptr;

    static int indexOf(char ch) { return ch == '{' ? 26 : ch - 'a'; }
};

int main() {
    WordFilter wf({"apple"});
    assert(wf.f("a", "e") == 0);
    assert(wf.f("a", "a") == -1);
    assert(wf.f("b", "") == -1);

    WordFilter wf2({"apple", "apply", "banana"});
    assert(wf2.f("a", "e") == 0);
    assert(wf2.f("a", "y") == 1);
    assert(wf2.f("ban", "na") == 2);
    assert(wf2.f("b", "e") == -1);

    WordFilter wf3({"ab", "b", "ab", "a", "ab"});
    assert(wf3.f("a", "b") == 4);
    std::cout << "word_filter: all tests passed\n";
    return 0;
}
