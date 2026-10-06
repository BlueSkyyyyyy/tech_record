"""45. 跳跃游戏 II（Jump Game II）

题目：给定一个非负整数数组 nums，你最初位于第一个下标，每个元素表示你在该位置可以跳跃的
最大长度。假设你总是可以到达最后一个下标，求到达最后一个下标的最少跳跃次数。

思路（贪心：按「层」推进，每层代表一次跳跃能达到的范围）：
    55 题只关心「能不能到」，本题还要「最少跳几步」。可以把它看成 BFS：
    第 0 层是起点，第 1 层是「跳 1 次能到达的所有位置」，第 2 层是「跳 2 次能到达的所有
    位置」，以此类推。同一层里的位置代价相同，所以每跨过一层的边界，跳跃次数就 +1。
    用两个变量：
      - cur_end：当前这一层能达到的最远下标（本层边界）；
      - farthest：在扫描本层时看到的、下一层能达到的最远下标。
    从左往右扫到倒数第二个位置（最后一个位置不用再跳）：
      - 每到一个位置，用 i + nums[i] 更新 farthest；
      - 当 i == cur_end 时，说明当前层已扫完，必须再跳一次才能进入下一层：jumps++，
        并把 cur_end 更新为 farthest。
    jumps 就是最少跳跃次数。

    为什么这样能得到最少次数：每一层都代表「用当前步数能到达的全部位置」，我们总是把
    下一层的边界扩张到最远。这样下一次跨层时，用同样的步数覆盖的范围最大，因此到达终点
    所需的层数（跳跃数）最少。这就是「最少步数 BFS」的贪心本质。

复杂度：时间 O(n)（一次遍历），空间 O(1)。
"""


def jump(nums):
    n = len(nums)
    if n <= 1:
        return 0
    jumps = 0
    cur_end = 0
    farthest = 0
    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])
        if i == cur_end:
            jumps += 1
            cur_end = farthest
    return jumps


if __name__ == "__main__":
    assert jump([2, 3, 1, 1, 4]) == 2
    assert jump([2, 3, 0, 1, 4]) == 2
    assert jump([0]) == 0
    assert jump([1, 2, 3]) == 2
    assert jump([2, 0, 1, 1, 4]) == 3
    print("jump_game_ii: all tests passed")
