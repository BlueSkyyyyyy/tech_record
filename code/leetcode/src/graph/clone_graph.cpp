// 133. 克隆图
// 见 clone_graph.py 的题目与思路说明。
#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <queue>
#include <vector>

class Node {
public:
    int val;
    std::vector<Node *> neighbors;
    Node() : val(0) {}
    explicit Node(int _val) : val(_val) {}
    Node(int _val, std::vector<Node *> _neighbors)
        : val(_val), neighbors(std::move(_neighbors)) {}
};

Node *dfs(Node *cur, std::map<Node *, Node *> &clones) {
    auto it = clones.find(cur);
    if (it != clones.end()) return it->second;
    Node *copy = new Node(cur->val);
    clones[cur] = copy;
    for (Node *nb : cur->neighbors) {
        copy->neighbors.push_back(dfs(nb, clones));
    }
    return copy;
}

Node *cloneGraph(Node *node) {
    if (node == nullptr) return nullptr;
    std::map<Node *, Node *> clones;
    return dfs(node, clones);
}

static std::vector<std::vector<int>> toAdj(Node *node) {
    std::vector<std::vector<int>> res;
    if (node == nullptr) return res;
    std::map<int, Node *> seen;
    std::vector<Node *> stack{node};
    while (!stack.empty()) {
        Node *cur = stack.back();
        stack.pop_back();
        if (seen.count(cur->val)) continue;
        seen[cur->val] = cur;
        for (Node *nb : cur->neighbors) {
            if (!seen.count(nb->val)) stack.push_back(nb);
        }
    }
    for (auto &kv : seen) {
        std::vector<int> nbrs;
        for (Node *nb : kv.second->neighbors) nbrs.push_back(nb->val);
        std::sort(nbrs.begin(), nbrs.end());
        res.push_back(nbrs);
    }
    return res;
}

int main() {
    // 构造 1-2-3-4-1 的环
    Node *n1 = new Node(1);
    Node *n2 = new Node(2);
    Node *n3 = new Node(3);
    Node *n4 = new Node(4);
    n1->neighbors = {n2, n4};
    n2->neighbors = {n1, n3};
    n3->neighbors = {n2, n4};
    n4->neighbors = {n1, n3};

    Node *cloned = cloneGraph(n1);
    assert(cloned != n1);

    std::vector<std::vector<int>> want = {{2, 4}, {1, 3}, {2, 4}, {1, 3}};
    assert(toAdj(cloned) == want);

    cloned->neighbors[0]->val = 99;
    assert(n1->neighbors[0]->val == 2);

    assert(cloneGraph(nullptr) == nullptr);

    Node *single = new Node(1);
    Node *one = cloneGraph(single);
    std::vector<std::vector<int>> want_single = {{}};
    assert(one != single && one->val == 1 && toAdj(one) == want_single);

    std::cout << "clone_graph: all tests passed\n";
    return 0;
}
