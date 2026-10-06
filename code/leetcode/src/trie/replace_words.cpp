// 648. 单词替换
// 见 replace_words.py 的题目与思路说明。
#include <array>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct TrieNode {
    std::array<TrieNode *, 26> child;
    bool isEnd = false;
    TrieNode() { child.fill(nullptr); }
};

std::string replaceWords(const std::vector<std::string> &dictionary, const std::string &sentence) {
    TrieNode *root = new TrieNode();
    for (const std::string &rootWord : dictionary) {
        TrieNode *node = root;
        for (char ch : rootWord) {
            int i = ch - 'a';
            if (!node->child[i]) node->child[i] = new TrieNode();
            node = node->child[i];
        }
        node->isEnd = true;
    }

    std::string result;
    std::istringstream iss(sentence);
    std::string word;
    bool first = true;
    while (iss >> word) {
        if (!first) result += ' ';
        first = false;
        TrieNode *node = root;
        std::string built;
        for (char ch : word) {
            if (node->isEnd) break;  // 已到某个词根，最短词根即它
            int i = ch - 'a';
            if (!node->child[i]) break;
            node = node->child[i];
            built += ch;
        }
        result += node->isEnd ? built : word;
    }
    return result;
}

int main() {
    std::vector<std::string> dictionary = {"cat", "bat", "rat"};
    assert(replaceWords(dictionary, "the cattle was rattled by the battery") ==
           "the cat was rat by the bat");

    std::vector<std::string> dict2 = {"a", "b", "c"};
    assert(replaceWords(dict2, "aadsfasf absbs bbab cadsfafs") == "a a b c");
    assert(replaceWords({"cat"}, "cat") == "cat");
    assert(replaceWords({}, "hello world") == "hello world");
    std::cout << "replace_words: all tests passed\n";
    return 0;
}
