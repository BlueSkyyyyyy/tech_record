// 239. 滑动窗口最大值（单调双端队列）
// 见 sliding_window_maximum.py 的题目与思路说明。
#include <cassert>
#include <deque>
#include <iostream>
#include <vector>

std::vector<int> maxSlidingWindow(const std::vector<int> &nums, int k) {
    std::deque<int> dq;  // 存下标，对应值单调递减
    std::vector<int> res;
    for (int i = 0; i < static_cast<int>(nums.size()); ++i) {
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);
        if (dq.front() <= i - k) dq.pop_front();
        if (i >= k - 1) res.push_back(nums[dq.front()]);
    }
    return res;
}

int main() {
    const std::vector<int> want1 = {3, 3, 5, 5, 6, 7};
    const std::vector<int> want2 = {1};
    const std::vector<int> want3 = {1, -1};
    const std::vector<int> want4 = {11};
    const std::vector<int> want5 = {12, 12, 12};
    assert(maxSlidingWindow({1, 3, -1, -3, 5, 3, 6, 7}, 3) == want1);
    assert(maxSlidingWindow({1}, 1) == want2);
    assert(maxSlidingWindow({1, -1}, 1) == want3);
    assert(maxSlidingWindow({9, 11}, 2) == want4);
    assert(maxSlidingWindow({4, 2, 12, 3, 5}, 3) == want5);
    std::cout << "sliding_window_maximum: all tests passed\n";
    return 0;
}
