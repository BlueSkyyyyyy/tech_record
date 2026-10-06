"""355. 设计推特（Design Twitter）

题目：设计一个简化版推特：
    postTweet(userId, tweetId)：用户发一条推文；
    getNewsFeed(userId)：返回该用户新闻流里最近的 10 条推文 id，最近的在前。新闻流 =
        该用户自己的推文 + 他关注的所有人的推文；
    follow(followerId, followeeId)：follower 关注 followee；
    unfollow(followerId, followeeId)：取消关注。

思路（哈希表 + 时间戳 + 多路归并取 Top-10）：
    新闻流要「按时间倒序取前 10」，涉及两类信息：谁的推文进流、推文的新旧顺序。

    - `tweets`：userId -> 该用户按时间递增排列的推文列表，每条记 `(时间戳, tweetId)`；
    - `following`：userId -> 关注的用户集合（用集合保证关注不重复、取关能删掉）；
    - 一个全局递增时间戳，保证不同用户的推文也能比较先后。

    `getNewsFeed` 把「自己 + 所有关注者」的推文列表当作若干条有序链做**多路归并**：
    每链从最新的末尾开始，用小顶堆（按时间戳取负）每次弹出最新的那条，再从同一条链里
    补上前一条，直到凑满 10 条或堆空。这样不需要把所有推文全排序，数据量大时也只碰
    每条链末尾的少量元素。

复杂度：postTweet / follow / unfollow 时间 O(1)；getNewsFeed 时间
O((F + 10) log F)（F 为关注的用户数），空间 O(总推文数)。
"""

import heapq


class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = {}       # user -> [(time, tweetId), ...]，时间递增
        self.following = {}    # user -> set(followee)

    def postTweet(self, userId, tweetId):
        self.time += 1
        self.tweets.setdefault(userId, []).append((self.time, tweetId))

    def getNewsFeed(self, userId):
        sources = [self.tweets.get(userId, [])]
        for followee in self.following.get(userId, ()):
            if followee != userId:
                sources.append(self.tweets.get(followee, []))

        heap = []
        for i, lst in enumerate(sources):
            if lst:
                heap.append((-lst[-1][0], i, len(lst) - 1))
        heapq.heapify(heap)

        feed = []
        while heap and len(feed) < 10:
            _, i, j = heapq.heappop(heap)
            feed.append(sources[i][j][1])
            if j > 0:
                heapq.heappush(heap, (-sources[i][j - 1][0], i, j - 1))
        return feed

    def follow(self, followerId, followeeId):
        self.following.setdefault(followerId, set()).add(followeeId)

    def unfollow(self, followerId, followeeId):
        if followerId in self.following:
            self.following[followerId].discard(followeeId)


if __name__ == "__main__":
    twitter = Twitter()
    twitter.postTweet(1, 5)
    assert twitter.getNewsFeed(1) == [5]
    twitter.follow(1, 2)
    twitter.postTweet(2, 6)
    assert twitter.getNewsFeed(1) == [6, 5]     # 关注者的新推文排在前面
    twitter.unfollow(1, 2)
    assert twitter.getNewsFeed(1) == [5]        # 取关后看不到 2 的推文

    for i in range(1, 13):                      # 发 12 条，验证只取最近 10 条
        twitter.postTweet(3, i)
    assert twitter.getNewsFeed(3) == [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]

    twitter.follow(4, 4)                        # 关注自己不应导致推文重复
    twitter.postTweet(4, 100)
    assert twitter.getNewsFeed(4) == [100]
    print("twitter: all tests passed")
