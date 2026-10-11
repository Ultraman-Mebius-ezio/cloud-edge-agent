"""预路由闸门：判断一条请求**值不值得拆**。

为什么需要它（v1 实验的直接结论）：DoT 原文对每个查询都走完整流水线，而
分解与汇聚是两次额外的云端调用。v1 实测这条流水线有一笔与题目难度**无关**的
固定开销——最省的一次请求也要 842 云端 token，而同一道题 direct 一个 token
都不用。根因就是简单请求也被无条件拆解。

所以闸门的作用是拦住两类请求，让它们走整请求直答：

1. **单步算术**：一个算式就能答完，拆了纯亏。
2. **开放式提问**：主观、无确定解，拆出的「子任务」彼此重叠，v1 实测这类题
   在 dot 下反而最慢（64~70 秒、9k~20k token）。

判定全部是**结构性信号**，不看题目难度、不看模型输出，因此零成本、确定性、
可复现（与 allocator 同一条设计原则：规则而非学习）。

**这是一个启发式，不是判决**。阈值都提成模块级常量，报告里会逐题公开每条
请求的闸门判定与理由，读者可以自己判断闸门拦得对不对。
"""

from __future__ import annotations

import re

from . import allocator

# ── 模块级常量：调参只改这里 ──────────────────────────────────────────────

# 出现这些词，说明题干显式写了求解顺序，几乎必然是多步。
# 注意这是**有序**元组，命中顺序确定，保证同一文本每次得到同一个理由。
SEQUENCE_MARKERS: tuple[str, ...] = (
    "然后",
    "接着",
    "之后",
    "最后",
    "再",
    "又",
    "连续",
    "分别",
    "各自",
    "先",
)

# 题干里出现这么多个数字，通常是多个已知量需要组合，属于多步题的强信号。
MULTI_NUMBER_THRESHOLD = 3

# 多个问号 = 多个待答子问题。
MULTI_QUESTION_THRESHOLD = 2

# 超长题干照搬 allocator 的阈值，避免两处各定一套。
LONG_TEXT_CHARS = allocator.LONG_TEXT_CHARS

_QUESTION_RE = re.compile(r"[?？]")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _count_numbers(text: str) -> int:
    return len(_NUMBER_RE.findall(text))


def _count_questions(text: str) -> int:
    return len(_QUESTION_RE.findall(text))


def _matched_sequence_marker(text: str) -> str | None:
    for word in SEQUENCE_MARKERS:
        if word in text:
            return word
    return None


def should_decompose(question: str) -> tuple[bool, str]:
    """返回 (是否拆解, 理由)。理由直接进报告，所以每个分支都要能自解释。

    判定顺序（先命中先返回，顺序本身是设计的一部分）：

    1. 纯算术语句 → 不拆（单步）
    2. 多步信号（顺序词 / 数字多 / 多问号 / 超长）→ 拆
    3. 开放性问题词 → 不拆（无确定解，拆了子任务互相重叠）
    4. 其余 → 不拆（短且无多步信号，视为单步）

    第 2 步排在第 3 步之前是有意的：有些题既含多步结构、又带「说明」这类
    开放词（例如「先算出…再减去…最后说明这个差值的含义」），它确实该拆，
    不能被开放词一票否决。
    """
    if allocator.is_single_step_arithmetic(question):
        return False, "单步算术语句"

    marker = _matched_sequence_marker(question)
    if marker is not None:
        return True, f"含顺序词「{marker}」"

    n_num = _count_numbers(question)
    if n_num >= MULTI_NUMBER_THRESHOLD:
        return True, f"含 {n_num} 个已知量"

    n_q = _count_questions(question)
    if n_q >= MULTI_QUESTION_THRESHOLD:
        return True, f"含 {n_q} 个问句"

    if len(question) > LONG_TEXT_CHARS:
        return True, f"长题干(>{LONG_TEXT_CHARS}字)"

    word = allocator.matched_open_word(question)
    if word is not None:
        return False, f"开放性问题（{word}）"

    return False, "短且无多步信号"


if __name__ == "__main__":
    # 自测：不触发任何 LLM 调用。
    _samples = [
        "37 乘以 48 等于多少？",
        "一箱苹果 24 个，吃掉 7 个还剩几个？",
        "一个班有 45 名学生，其中 3/5 是女生。女生比男生多几人？",
        "小明每天跑步 3 公里，一周跑 5 天，连续跑 4 周，总共跑了多少公里？",
        "请解释端云协同架构的主要优势和潜在风险，并比较它与纯云端方案的差异。",
        "为什么在移动端部署大模型时，量化通常比蒸馏更划算？请分析原因。",
        "先算出 18 的平方是多少，再用这个结果减去 100，最后说明这个差值正负号的物理含义。",
    ]
    for s in _samples:
        need, why = should_decompose(s)
        print(f"{'拆' if need else '不拆'}  {why:22s} {s[:30]}")
