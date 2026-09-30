// 496. 下一个更大元素 I
// 见 next_greater_element_i.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <vector>

std::vector<int> nextGreaterElement(const std::vector<int> &nums1,
                                    const std::vector<int> &nums2) {
    std::unordered_map<int, int> nextGreater;
    std::stack<int> st;
    for (int x : nums2) {
        while (!st.empty() && st.top() < x) {
            nextGreater[st.top()] = x;
            st.pop();
        }
        st.push(x);
    }
    std::vector<int> result;
    for (int x : nums1) {
        auto it = nextGreater.find(x);
        result.push_back(it == nextGreater.end() ? -1 : it->second);
    }
    return result;
}

int main() {
    std::vector<int> want1 = {-1, 3, -1};
    std::vector<int> want2 = {3, -1};
    std::vector<int> want3 = {7, 7, 7};
    assert(nextGreaterElement({4, 1, 2}, {1, 3, 4, 2}) == want1);
    assert(nextGreaterElement({2, 4}, {1, 2, 3, 4}) == want2);
    assert(nextGreaterElement({1, 3, 5}, {6, 5, 4, 3, 2, 1, 7}) == want3);
    std::cout << "next_greater_element_i: all tests passed\n";
    return 0;
}
