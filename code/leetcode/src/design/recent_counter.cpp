// 933. 最近的请求次数
// 见 recent_counter.py 的题目与思路说明。
#include <cassert>
#include <deque>
#include <iostream>

class RecentCounter {
public:
    int ping(int t) {
        q_.push_back(t);
        while (q_.front() < t - 3000) {
            q_.pop_front();
        }
        return static_cast<int>(q_.size());
    }

private:
    std::deque<int> q_;
};

int main() {
    RecentCounter counter;
    assert(counter.ping(1) == 1);
    assert(counter.ping(100) == 2);
    assert(counter.ping(3001) == 3);
    assert(counter.ping(3002) == 3);   // 1 已过期
    assert(counter.ping(7000) == 1);   // 只剩 7000 自己
    assert(counter.ping(7001) == 2);

    std::cout << "recent_counter: all tests passed\n";
    return 0;
}
