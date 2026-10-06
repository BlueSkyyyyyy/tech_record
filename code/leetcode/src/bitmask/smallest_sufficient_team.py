"""1125. 最小的必要团队（Smallest Sufficient Team）

题目：给定技能列表 req_skills 和每个人的技能 people[i]，选出人数最少的团队，使其
技能的并集覆盖所有 req_skills。

思路（状态 = "已经覆盖的技能集合"）：
    每种技能只有"被覆盖 / 没被覆盖"两种状态，m 种技能就是 2^m 个集合，用掩码表示。
    把每个人的技能也压成掩码。于是问题变成：从这些"人的掩码"里选最少几个，使按位或
    等于全集。这是一个在"可达集合"上的动态规划/广度扩展。

    dp[mask] 记录"覆盖情况为 mask 时，所选的人下标列表"。初始 dp[0] = []。每考虑一个
    新人 p，把它接到所有已有集合后面：如果并集新增了技能（或人数更少）就更新。

    这里把"人的下标列表"直接存进状态，是为了最后能输出具体是哪些人；n 和 m 都不超过
    16，列表开销可以接受。若只求最小人数，存"人数"即可。

复杂度：时间 O(n * 2^m)，空间 O(2^m * n)（存列表）。
"""


def smallest_sufficient_team(req_skills, people):
    m = len(req_skills)
    skill_id = {skill: i for i, skill in enumerate(req_skills)}

    people_mask = []
    for skills in people:
        mask = 0
        for skill in skills:
            mask |= 1 << skill_id[skill]
        people_mask.append(mask)

    full = (1 << m) - 1
    dp = {0: ()}
    for i, mask in enumerate(people_mask):
        for covered, team in list(dp.items()):
            merged = covered | mask
            if merged == covered:
                continue
            cand = team + (i,)
            if merged not in dp or len(dp[merged]) > len(cand):
                dp[merged] = cand
    return list(dp[full])


if __name__ == "__main__":
    assert smallest_sufficient_team(
        ["java", "nodejs", "reactjs"],
        [["java"], ["nodejs"], ["nodejs", "reactjs"]],
    ) == [0, 2]

    assert sorted(smallest_sufficient_team(
        ["algorithms", "math", "java", "reactjs", "csharp", "aws"],
        [
            ["algorithms", "math", "java"],
            ["algorithms", "math", "reactjs"],
            ["java", "csharp", "aws"],
            ["reactjs", "csharp"],
            ["csharp", "math"],
            ["aws", "java"],
        ],
    )) == [1, 2]

    assert smallest_sufficient_team(["c", "cpp"], [["c", "cpp"]]) == [0]
    print("smallest_sufficient_team: all tests passed")
