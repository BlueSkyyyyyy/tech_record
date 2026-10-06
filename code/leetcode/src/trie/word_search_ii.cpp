// 212. 单词搜索 II
// 见 word_search_ii.py 的题目与思路说明。
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

struct TrieNode {
    std::array<TrieNode *, 26> child;
    std::string word;
    TrieNode() { child.fill(nullptr); }
};

class WordSearchII {
  public:
    std::vector<std::string> findWords(std::vector<std::vector<char>> &board,
                                       const std::vector<std::string> &words) {
        TrieNode *root = new TrieNode();
        for (const std::string &word : words) {
            TrieNode *node = root;
            for (char ch : word) {
                int i = ch - 'a';
                if (!node->child[i]) node->child[i] = new TrieNode();
                node = node->child[i];
            }
            node->word = word;
        }

        result_.clear();
        rows_ = static_cast<int>(board.size());
        cols_ = static_cast<int>(board[0].size());
        for (int r = 0; r < rows_; ++r)
            for (int c = 0; c < cols_; ++c) dfs(board, r, c, root);
        return result_;
    }

  private:
    std::vector<std::string> result_;
    int rows_ = 0;
    int cols_ = 0;

    void dfs(std::vector<std::vector<char>> &board, int r, int c, TrieNode *node) {
        char ch = board[r][c];
        TrieNode *nxt = node->child[ch - 'a'];
        if (!nxt) return;
        if (!nxt->word.empty()) {
            result_.push_back(nxt->word);
            nxt->word.clear();  // 置空，避免同一单词被重复收集
        }
        board[r][c] = '#';
        const int dr[4] = {1, -1, 0, 0};
        const int dc[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k];
            int nc = c + dc[k];
            if (nr >= 0 && nr < rows_ && nc >= 0 && nc < cols_ && board[nr][nc] != '#')
                dfs(board, nr, nc, nxt);
        }
        board[r][c] = ch;
    }
};

int main() {
    std::vector<std::vector<char>> board = {
        {'o', 'a', 'a', 'n'}, {'e', 't', 'a', 'e'}, {'i', 'h', 'k', 'r'}, {'i', 'f', 'l', 'v'}};
    std::vector<std::string> words = {"oath", "pea", "eat", "rain"};
    WordSearchII solver;
    std::vector<std::string> got = solver.findWords(board, words);
    std::sort(got.begin(), got.end());
    std::vector<std::string> want = {"eat", "oath"};
    assert(got == want);

    std::vector<std::vector<char>> board2 = {{'a', 'a'}};
    std::vector<std::string> words2 = {"a", "a"};
    std::vector<std::string> got2 = solver.findWords(board2, words2);
    std::vector<std::string> want2 = {"a"};
    assert(got2 == want2);

    std::cout << "word_search_ii: all tests passed\n";
    return 0;
}
