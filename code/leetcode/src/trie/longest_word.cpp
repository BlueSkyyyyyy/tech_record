// 720. 词典中最长的单词
// 见 longest_word.py 的题目与思路说明。
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

class LongestWord {
  public:
    std::string longestWord(const std::vector<std::string> &words) {
        TrieNode *root = new TrieNode();
        for (const std::string &word : words) {
            TrieNode *node = root;
            for (char ch : word) {
                int i = ch - 'a';
                if (!node->child[i]) node->child[i] = new TrieNode();
                node = node->child[i];
            }
            node->isEnd = true;
        }
        std::string best;
        dfs(root, "", best);
        return best;
    }

  private:
    void dfs(TrieNode *node, const std::string &path, std::string &best) {
        if (node->isEnd) {
            if (path.size() > best.size() ||
                (path.size() == best.size() && path < best))
                best = path;
        }
        for (int i = 0; i < 26; ++i) {
            TrieNode *child = node->child[i];
            if (child && child->isEnd) dfs(child, path + char('a' + i), best);
        }
    }
};

int main() {
    LongestWord solver;
    std::vector<std::string> w1 = {"w", "wo", "wor", "worl", "world"};
    assert(solver.longestWord(w1) == "world");

    std::vector<std::string> w2 = {"a", "banana", "app", "appl",
                                   "ap", "apply", "apple"};
    assert(solver.longestWord(w2) == "apple");

    std::vector<std::string> w3 = {"ab", "a", "abc", "abd"};
    assert(solver.longestWord(w3) == "abc");

    std::vector<std::string> w4 = {"b", "ba", "ban", "bandanas", "bandana"};
    assert(solver.longestWord(w4) == "ban");

    std::vector<std::string> w5 = {"xy", "xyz"};
    assert(solver.longestWord(w5) == "");
    std::cout << "longest_word: all tests passed\n";
    return 0;
}
