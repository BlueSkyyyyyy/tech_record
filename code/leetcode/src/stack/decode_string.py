"""394. 字符串解码（Decode String）

题目：给定一个编码字符串，形如 k[encoded_string]，表示方括号里的内容重复 k 次。
      可以嵌套，如 3[a2[c]] 表示 a2[c] 先解码成 acc，再整体重复 3 次得 accaccacc。
      返回解码后的字符串。

思路（两个栈，一个存「外层已拼好的串」，一个存重复次数）：
    从左到右扫描：
      - 遇到数字：累积当前数字 num（可能多位，用 num = num*10 + digit）；
      - 遇到 '['：把「当前已拼好的串 cur」和「当前重复次数 num」一起压栈，
        然后清空 cur、num，开始处理括号里的内容；
      - 遇到 ']'：弹出外层串 prev 和次数 repeat，把括号内的 cur 重复 repeat 次，
        再拼回 prev，作为新的 cur；
      - 遇到字母：直接接到 cur 后面。

    为什么用栈：解码是从里到外的——必须先解出最内层方括号，才能去重复外层。
    但扫描是从左到右、先遇到外层的 '['。栈正好把这个顺序倒过来：
    '[' 时把外层的上下文暂存，']' 时取回，天然匹配「后进先出」的嵌套结构。

    为什么数字要单独用累加而不是直接 int：因为 k 可能是多位数（如 100[leetcode]），
    必须等遇到 '[' 才知道这个数字读完了。

复杂度：时间 O(解码后串长)——每个字符被处理的次数与其最终出现次数同一量级；
      空间 O(解码后串长) 用于存放中间结果。
"""


def decode_string(s):
    stack = []
    cur = ""
    num = 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == "[":
            stack.append((cur, num))
            cur = ""
            num = 0
        elif ch == "]":
            prev, repeat = stack.pop()
            cur = prev + cur * repeat
        else:
            cur += ch
    return cur


if __name__ == "__main__":
    assert decode_string("3[a]2[bc]") == "aaabcbc"
    assert decode_string("3[a2[c]]") == "accaccacc"
    assert decode_string("2[abc]3[cd]ef") == "abcabccdcdcdef"
    assert decode_string("abc") == "abc"
    print("decode_string: all tests passed")
