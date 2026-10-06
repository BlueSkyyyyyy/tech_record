// 355. 设计推特
// 见 twitter.py 的题目与思路说明。
#include <cassert>
#include <iostream>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class Twitter {
public:
    void postTweet(int userId, int tweetId) {
        ++time_;
        tweets_[userId].emplace_back(time_, tweetId);
    }

    std::vector<int> getNewsFeed(int userId) {
        std::vector<const std::vector<std::pair<int, int>> *> sources;
        auto addSource = [&](int uid) {
            auto it = tweets_.find(uid);
            if (it != tweets_.end() && !it->second.empty()) {
                sources.push_back(&it->second);
            } else {
                sources.push_back(nullptr);
            }
        };
        addSource(userId);
        auto fit = following_.find(userId);
        if (fit != following_.end()) {
            for (int followee : fit->second) {
                if (followee != userId) {
                    addSource(followee);
                }
            }
        }

        std::priority_queue<std::tuple<int, int, int>> pq;
        for (int i = 0; i < static_cast<int>(sources.size()); ++i) {
            if (sources[i]) {
                int j = static_cast<int>(sources[i]->size()) - 1;
                pq.emplace((*sources[i])[j].first, i, j);
            }
        }

        std::vector<int> feed;
        while (!pq.empty() && static_cast<int>(feed.size()) < 10) {
            auto [ts, i, j] = pq.top();
            pq.pop();
            feed.push_back((*sources[i])[j].second);
            if (j > 0) {
                pq.emplace((*sources[i])[j - 1].first, i, j - 1);
            }
        }
        return feed;
    }

    void follow(int followerId, int followeeId) {
        following_[followerId].insert(followeeId);
    }

    void unfollow(int followerId, int followeeId) {
        auto it = following_.find(followerId);
        if (it != following_.end()) {
            it->second.erase(followeeId);
        }
    }

private:
    int time_ = 0;
    std::unordered_map<int, std::vector<std::pair<int, int>>> tweets_;  // user -> (time, id)
    std::unordered_map<int, std::unordered_set<int>> following_;        // user -> followees
};

int main() {
    Twitter twitter;
    twitter.postTweet(1, 5);
    std::vector<int> want1 = {5};
    assert(twitter.getNewsFeed(1) == want1);
    twitter.follow(1, 2);
    twitter.postTweet(2, 6);
    std::vector<int> want2 = {6, 5};
    assert(twitter.getNewsFeed(1) == want2);
    twitter.unfollow(1, 2);
    assert(twitter.getNewsFeed(1) == want1);

    for (int i = 1; i <= 12; ++i) {           // 发 12 条，验证只取最近 10 条
        twitter.postTweet(3, i);
    }
    std::vector<int> want3 = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3};
    assert(twitter.getNewsFeed(3) == want3);

    twitter.follow(4, 4);                     // 关注自己不应导致推文重复
    twitter.postTweet(4, 100);
    std::vector<int> want4 = {100};
    assert(twitter.getNewsFeed(4) == want4);

    std::cout << "twitter: all tests passed\n";
    return 0;
}
