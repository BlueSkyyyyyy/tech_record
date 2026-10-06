// 1268. 搜索推荐系统
// 见 suggested_products.py 的题目与思路说明。
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

struct TrieNode {
    std::array<TrieNode *, 26> child;
    std::vector<std::string> suggest;
    TrieNode() { child.fill(nullptr); }
};

class Solution {
  public:
    std::vector<std::vector<std::string>>
    suggestedProducts(std::vector<std::string> products, const std::string &searchWord) {
        std::sort(products.begin(), products.end());
        TrieNode *root = new TrieNode();
        for (const std::string &product : products) {
            TrieNode *node = root;
            for (char ch : product) {
                int i = ch - 'a';
                if (!node->child[i]) node->child[i] = new TrieNode();
                node = node->child[i];
                if (node->suggest.size() < 3) node->suggest.push_back(product);
            }
        }

        std::vector<std::vector<std::string>> result;
        TrieNode *node = root;
        for (char ch : searchWord) {
            if (node) node = node->child[ch - 'a'];
            result.push_back(node ? node->suggest : std::vector<std::string>{});
        }
        return result;
    }
};

int main() {
    Solution solver;

    std::vector<std::string> p1 = {"mobile", "mouse", "moneypot", "monitor",
                                   "mousepad"};
    std::vector<std::vector<std::string>> want1 = {
        {"mobile", "moneypot", "monitor"},
        {"mobile", "moneypot", "monitor"},
        {"mouse", "mousepad"},
        {"mouse", "mousepad"},
        {"mouse", "mousepad"}};
    assert(solver.suggestedProducts(p1, "mouse") == want1);

    std::vector<std::string> p2 = {"havana"};
    std::vector<std::vector<std::string>> want2 = {
        {"havana"}, {"havana"}, {"havana"},
        {"havana"}, {"havana"}, {"havana"}};
    assert(solver.suggestedProducts(p2, "havana") == want2);

    std::vector<std::string> p3 = {"bags", "baggage", "banner", "box", "cloths"};
    std::vector<std::vector<std::string>> want3 = {
        {"baggage", "bags", "banner"},
        {"baggage", "bags", "banner"},
        {"baggage", "bags"},
        {"bags"}};
    assert(solver.suggestedProducts(p3, "bags") == want3);
    std::cout << "suggested_products: all tests passed\n";
    return 0;
}
