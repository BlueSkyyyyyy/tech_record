"""1011. 在 D 天内送达包裹的能力（Capacity To Ship Packages Within D Days）

题目：包裹必须**按给定顺序**装上船，在 days 天内全部运完。船每天可装载一个连续区间的包裹，
      总重量不能超过船的载重 capacity。求能在 days 天内运完所有包裹的**最小载重**。

思路：又是**二分答案**：不去枚举怎么切分，而是猜载重 capacity，验证它能不能在 days 天内运完。
      单调性同样成立——载重越大，需要天数越少，所以「可行」随载重增大由假变真，二分临界点即可。

      验证函数 can_ship(cap)：贪心地按顺序装，能用当前船装下就继续装，装不下就开新的一天。
      得到的「使用天数 <= days」则可行。

      答案区间：下界是 max(weights)（载重再小就装不下最重的单个包裹），
      上界是 sum(weights)（一天全运走，必然可行）。二分第一个可行的载重。

      为什么这道题和 875 是同一套模板：两者都是「最小化一个阈值，使某个可行性判定成立」，
      区别只在验证函数怎么写。所以这类题只要抓住「猜答案 + 单调验证」两步即可。

复杂度：时间 O(n log(sum(weights)))，空间 O(1)。
"""


def ship_within_days(weights, days):
    def can_ship(cap):
        used_days = 1
        cur = 0
        for w in weights:
            if cur + w > cap:
                used_days += 1
                cur = 0
            cur += w
        return used_days <= days

    left, right = max(weights), sum(weights)
    while left < right:
        mid = left + (right - left) // 2
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1
    return left


if __name__ == "__main__":
    assert ship_within_days([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5) == 15
    assert ship_within_days([3, 2, 2, 4, 1, 4], 3) == 6
    assert ship_within_days([1, 2, 3, 1, 1], 4) == 3
    assert ship_within_days([10], 1) == 10
    assert ship_within_days([5, 5, 5], 3) == 5
    print("capacity_to_ship_packages_within_d_days: all tests passed")
