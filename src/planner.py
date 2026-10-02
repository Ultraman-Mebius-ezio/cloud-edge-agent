"""问题分解（planner）——DoT 流水线的第 ① 步。

职责：把一个用户问题拆成若干「可独立求解」的子任务，交给下游的
allocator（定档位）与 runner（逐条执行）。本模块**只负责拆**，
不判断走云还是走本地——那是 allocator 的规则职责，混在一起会让
「子任务级分配」这条卖点说不清。

为什么提示词是中文重写，而不是照译 DoT 论文附录 C.1：
论文的 8 个示例是英文（GSM8K / 常识推理），直接翻译过来会失真。
本机实测（见接口冻结 §5）3B 模型在中文场景下会产出「复述题干」的
伪子任务，例如把「每层能放几本书」当成一个步骤——它没有推进任何求解，
只是把题干条件换个说法抄了一遍。所以中文版提示词必须：
  1. 用中文 few-shot 示例把「可独立求解、后一步引用前一步结果」示范出来；
  2. 点名禁止复述题干、禁止解释文字、禁止无关步骤。
"""

from __future__ import annotations

import re
import sys

from .llm_client import call_llm

# --------------------------------------------------------------------------
# 模块级常量：所有阈值集中在这里，便于以后扫描/调参。
# 散落的魔法数字会让「拆解粒度」变成不可控变量，实验复现不了。
# --------------------------------------------------------------------------

# 子任务数上限。超过就截断——DoT 的子任务越多，云端调用与 token 越线性膨胀，
# 而 3B 执行器在很长的依赖链上会累积错误。8 是「够拆 GSM8K 级别的多步题、
# 又不至于把规划开销吃掉」的折中值。
MAX_SUBTASKS = 8

# 少于 2 条视为「没拆出东西」：一条等于没拆，直接退化为 [question]。
MIN_SUBTASKS = 2

# 原始输出留存长度。分解失败时把模型原话截断打出来，方便排查是
# 「模型没按格式输出」还是「调用本身就失败了」——这两种要分开归因。
RAW_TEXT_LOG_CHARS = 500

# 分解调用的固定参数。抽成常量是为了让 CLI/实验脚本一眼看到口径。
DECOMPOSER_TIER = "cloud"
DECOMPOSER_STRATEGY = "dot"
DECOMPOSER_RETRIES = 2


# 解析用正则。第一层：行首「数字 + 分隔符」，如 "1. " / "2、" / "3)" / "4）"。
_NUMBERED_RE = re.compile(r"^\s*\d+\s*[.、)）]\s*(.+)$")
# 第二层兜底用的项目符号：- * • · 后跟空白。
_BULLET_RE = re.compile(r"^\s*[-*•·]\s+(.+)$")
# markdown 强调符：模型偶尔会用 **加粗** 或 `代码` 包住子任务。
_EMPHASIS_CHARS = "*`#"


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------

# 系统提示词按 DoT 附录 C.1 的骨架改写为中文。
# 骨架：「我给你一个问题类型 → 请拆成若干易解步骤 → 示例如下 → 现在给你
# 具体问题 → 按示例格式拆解」。这里把「示例」和「格式要求」放进 system，
# 把「具体问题」放进 user，职责更清晰。
_SYSTEM_PROMPT = """你是一个问题分解器。你的唯一职责是：把一个复杂问题拆成若干个「可独立求解」的小步骤，供后续逐步求解。后一个步骤可以直接使用前面步骤已经算出的结果。

【输出格式】
- 每个子步骤单独占一行，行首用阿拉伯数字加英文句点编号，例如：
1. 第一步要做的事
2. 第二步要做的事
- 只输出编号和步骤本身。除了这些行，不要输出任何其它内容。

【绝对禁止】
- 禁止复述、改写、概括原题。把题干换一种说法再抄一遍，不构成一个步骤。
- 禁止输出解释、说明、铺垫、开场白、总结，或「以下是分解」之类的文字。
- 禁止输出与解题无关的步骤，例如「看看每层能放几本书」这类只是重复题干条件的伪步骤。
- 禁止在步骤里直接写出最终答案。

【示例 1】
问题：小明买了 3 本书，每本 12 元，又买了一个 25 元的笔记本，付给收银员 100 元，应找回多少元？
1. 计算 3 本书的总价，即 3 乘以 12。
2. 计算所有商品的总价，即第 1 步的结果加上笔记本的 25 元。
3. 计算应找回的金额，即 100 元减去第 2 步得到的总价。

【示例 2】
问题：2022 年卡塔尔世界杯的冠军是哪支球队？该球队所在大洲的人口是否超过 1 亿？
1. 查明 2022 年卡塔尔世界杯的冠军球队。
2. 查明第 1 步得到的球队所在的大洲。
3. 查明第 2 步所确定大洲的总人口数。
4. 把第 3 步得到的人口数与 1 亿作比较，得出结论。

【示例 3】
问题：一个班有 45 人，其中 3/5 是女生，女生比男生多几人？
1. 计算女生人数，即 45 乘以 3/5。
2. 计算男生人数，即 45 减去第 1 步得到的女生人数。
3. 计算女生比男生多出的数量，即第 1 步的女生人数减去第 2 步的男生人数。"""


def _user_prompt(question: str) -> str:
    """把具体问题作为「命令」发给模型，示例已在上面的 system 里。"""
    return (
        f"问题：{question}\n\n"
        "请参照上面的示例，把这个问题拆成若干个容易求解的步骤。"
        "每个步骤单独占一行，以编号开头（如 “1. ...”）。"
    )


# --------------------------------------------------------------------------
# 解析（三层降级）
# --------------------------------------------------------------------------

def _clean_subtask(s: str) -> str:
    """去掉首尾空白与 markdown 强调符。

    模型常把子任务写成 "**计算总价**" 或 "`第一步`"；
    这些符号留在文本里会污染 allocator 的长度判定与词表命中，所以在这里剥掉。
    """
    return s.strip().strip(_EMPHASIS_CHARS).strip()


def _parse_subtasks(text: str) -> list[str]:
    """把模型输出解析成子任务列表，按契约做三层降级。

    返回可能为空列表——空表示「没解析出可用子任务」，由调用方决定
    退化为 [question]。这里不抛异常，故障排查靠调用方打印原始输出。

    三层顺序（接口冻结 §2）：
      1) 按行首编号抽行；
      2) 抽不到就按换行切分，去掉行首项目符号；
      3) 仍不行 → 返回 []，调用方退化为 [question]。
    """
    lines = (text or "").splitlines()

    # 第一层：正常输出应当只走到这一层。先剥掉行首的 markdown 强调符再匹配
    # 编号——模型有时会输出 "**1. xxx**"，不先清洗就会漏掉整行编号。
    numbered: list[str] = []
    for line in lines:
        m = _NUMBERED_RE.match(_clean_subtask(line))
        if m:
            cleaned = _clean_subtask(m.group(1))
            if cleaned:
                numbered.append(cleaned)
    if numbered:
        return numbered

    # 第二层：没有编号。整段只有一行说明模型把结果写成了连续段落，
    # 换行切分没有意义，直接交给第三层退化。
    if len(lines) <= 1:
        return []
    fallback: list[str] = []
    for line in lines:
        stripped = line
        mb = _BULLET_RE.match(stripped)
        if mb:
            stripped = mb.group(1)
        cleaned = _clean_subtask(stripped)
        if cleaned:
            fallback.append(cleaned)
    return fallback


# --------------------------------------------------------------------------
# 对外接口
# --------------------------------------------------------------------------

def _log_raw(text: str, reason: str) -> None:
    """分解失败/异常时把模型原话截断打到 stderr，便于区分失败类型。"""
    snippet = (text or "").replace("\n", " ⏎ ")
    print(
        f"[planner] {reason}；原始输出（{len(text or '')} 字，截断 {RAW_TEXT_LOG_CHARS}）：{snippet[:RAW_TEXT_LOG_CHARS]}",
        file=sys.stderr,
    )


def decompose(question: str, *, task_id: str = "", run_id: str = "default") -> list[str]:
    """把问题拆成若干可独立求解的子任务。

    失败一律退化为 `[question]`（即「不拆」）——**这是合法结果，不是错误**。
    上游 runner 会照常按单条子任务执行，整条流水线不会因为分解失败而中断。
    契约要求的所有数值口径（子任务上限、最少条数、原始输出留存长度）
    都在上面的模块级常量里，不在这里散落魔法数字。
    """
    try:
        result = call_llm(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(question)},
            ],
            role="decomposer",
            tier=DECOMPOSER_TIER,
            strategy=DECOMPOSER_STRATEGY,
            task_id=task_id,
            run_id=run_id,
            retries=DECOMPOSER_RETRIES,
        )
    except Exception as e:  # noqa: BLE001
        # call_llm 正常不抛（网络故障会以 result.error 返回），但分解失败
        # 绝不能把整条流水线带崩，所以这里再兜一层。
        print(f"[planner] 分解调用异常，退化为不拆：{type(e).__name__}: {e}", file=sys.stderr)
        return [question]

    if result.error:
        # 契约明确要求：云端调用失败 → [question] + error 打到 stderr。
        print(f"[planner] 分解调用失败，退化为不拆：{result.error}", file=sys.stderr)
        _log_raw(result.text, "失败调用留存")
        return [question]

    subtasks = _parse_subtasks(result.text)

    if len(subtasks) > MAX_SUBTASKS:
        print(
            f"[planner] 子任务数 {len(subtasks)} 超过上限 {MAX_SUBTASKS}，已截断。",
            file=sys.stderr,
        )
        subtasks = subtasks[:MAX_SUBTASKS]

    if len(subtasks) < MIN_SUBTASKS:
        _log_raw(result.text, f"仅解析出 {len(subtasks)} 条子任务（少于 {MIN_SUBTASKS}），退化为不拆")
        return [question]

    return subtasks


if __name__ == "__main__":
    # 自测解析层：不触发任何 LLM 调用，纯格式验证。
    _samples = [
        ("1. 计算女生人数\n2. 计算男生人数\n3. 求差", ["计算女生人数", "计算男生人数", "求差"]),
        ("**1. 第一步**\n2) 第二步\n3、第三步", ["第一步", "第二步", "第三步"]),
        ("- 甲\n- 乙\n- 丙", ["甲", "乙", "丙"]),
        ("只有一个段落没有换行", []),
        ("", []),
    ]
    for raw, expect in _samples:
        got = _parse_subtasks(raw)
        ok = "OK " if got == expect else "FAIL"
        print(f"{ok} {raw!r} -> {got}")
