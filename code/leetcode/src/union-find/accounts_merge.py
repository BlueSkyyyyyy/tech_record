"""721. 账户合并（Accounts Merge）

题目：每个账户是 [名字, 邮箱1, 邮箱2, ...]。不同账户只要共享任一个邮箱就是同一个人
（邮箱还会传递合并）。把属于同一个人的账户合并：输出 [名字, 所有邮箱按字典序排序]。
结果顺序任意。

思路（邮箱做点，账户身份靠并查集合并）：
    名字不能当身份（重名很常见），真正的连接点是邮箱——两个账户共享一个邮箱就是同一个人。
    步骤：
    1. 用哈希表 `email_to_id` 记录每个邮箱「第一次出现在哪个账户」；
    2. 遍历每个账户的每个邮箱：若该邮箱之前出现过，说明当前账户与它所属账户是同一个人，
       把两个账户在并查集里 union；
    3. 按「根账户」把邮箱收拢起来；
    4. 每个根账户输出 [根账户的名字] + 排序后的邮箱列表。

    为什么邮箱没出现时记录的是账户下标而不是邮箱自己：并查集的元素是「账户」，
    邮箱只是用来发现「哪些账户该合并」的线索。

复杂度：时间 O(E·α(A) + E log E)（E 为邮箱总数，A 为账户数；排序是主要开销），空间 O(E)。
"""
from collections import defaultdict

from dsu import DSU


def accounts_merge(accounts):
    email_to_id = {}
    dsu = DSU(len(accounts))
    for i, account in enumerate(accounts):
        for email in account[1:]:
            if email in email_to_id:
                dsu.union(i, email_to_id[email])
            else:
                email_to_id[email] = i

    root_to_emails = defaultdict(list)
    for email, i in email_to_id.items():
        root_to_emails[dsu.find(i)].append(email)

    result = []
    for root, emails in root_to_emails.items():
        result.append([accounts[root][0]] + sorted(emails))
    result.sort(key=lambda item: item[1])
    return result


if __name__ == "__main__":
    accounts = [
        ["John", "johnsmith@mail.com", "john_newyork@mail.com"],
        ["John", "johnsmith@mail.com", "john00@mail.com"],
        ["Mary", "mary@mail.com"],
        ["John", "johnnybravo@mail.com"],
    ]
    want = [
        ["John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"],
        ["John", "johnnybravo@mail.com"],
        ["Mary", "mary@mail.com"],
    ]
    assert accounts_merge(accounts) == want
    assert accounts_merge([["Gabe", "g@m.com"]]) == [["Gabe", "g@m.com"]]
    print("accounts_merge: all tests passed")
