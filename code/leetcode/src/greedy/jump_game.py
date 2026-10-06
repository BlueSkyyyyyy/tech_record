"""55. 跳跃游戏（Jump Game）

题目：给定一个非负整数数组 nums，你最初位于数组的第一个下标。数组中的每个元素代表你在
该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标。

思路（贪心：只维护「能到达的最远位置」）：
    从左往右扫，用一个变量 reach 记录「当前能够到达的最远下标」。每走到位置 i：
      - 如果 i > reach，说明位置 i 根本走不到，更到不了终点，直接返回 False；
      - 否则用 i + nums[i] 更新 reach（从 i 再往前能延伸到的最远处）。
    遍历结束（能走到最后）就返回 True。

    为什么不需要关心「具体怎么跳」：能不能到达某个位置，只取决于「它是否在可达范围内」。
    只要当前位置 i 可达，从它出发就能把可达右边界推到 i + nums[i]；我们不断把右边界往右
    推，能推到哪里就代表这些位置都可达。因此只需维护这个最远可达点，跳跃的具体路径无关紧要。
    注意：在可达范围内每个位置都要尝试扩展右边界，所以是「能跳多远就更新多远」，而不是
    只从某个固定点起跳。

复杂度：时间 O(n)（一次遍历），空间 O(1)。
"""


def can_jump(nums):
    reach = 0
    for i, step in enumerate(nums):
        if i > reach:
            return False
        reach = max(reach, i + step)
    return True


if __name__ == "__main__":
    assert can_jump([2, 3, 1, 1, 4]) is True
    assert can_jump([3, 2, 1, 0, 4]) is False
    assert can_jump([0]) is True
    assert can_jump([2, 0, 0]) is True
    assert can_jump([1, 0, 1]) is False
    assert can_jump([0, 1]) is False
    print("jump_game: all tests passed")
