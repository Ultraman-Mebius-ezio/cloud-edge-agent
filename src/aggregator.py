"""汇聚器：把各子任务的答案合成一个面向用户的最终答案。

为什么要单独一个模块、单独一次 LLM 调用：DoT 的收益点之一就是「拆解 → 执行 → 汇聚」，
而汇聚这一步本身就是**规划开销**的一部分。metrics.plan_overhead 把 role='aggregator'
单独拎出来算 token 与耗时，所以它不能塞进 runner——否则规划开销会被混进执行开销，
报告里就拆不开「拆解本身值不值」这笔账了。

下面三条防御分支都是刻意的，不是过度设计：

1. 全失败不调 LLM —— 一次必然无效的云端调用，不仅白花钱，还会把「失败」记成
   一次成功的规划调用，污染 token 统计。
2. 汇聚失败降级拼接 —— 用户该拿到的是答案，而不是一个异常或空串。
3. prompt 长度保护 —— 这里请求的是 tier="cloud"，但云端没配 key 时 llm_client
   会**静默降级到本地**（窗口只有 16384）。不设预算，子任务一多就会超窗被截断，
   而超窗截断不报错——属于最难发现的一类指标污染。
"""

from __future__ import annotations

import sys

from .config import local_prompt_budget
from .llm_client import call_llm
from .runner import SubtaskResult

# ── 模块级常量：所有阈值集中在这里，方便后续扫描 / 调参，不散落魔法数字 ──

# 单个答案进聚合 prompt 的字符上限。子任务答案动辄上千字，全量塞进去会让长尾
# 答案挤掉其它子任务；截断保留的上下文比整条丢弃更多。
MAX_ANSWER_CHARS = 1200

# 失败原因写进 prompt 的截断长度。异常信息可能很长，但模型只需要知道「这条失败了」。
MAX_ERROR_CHARS = 200

# 聚合 prompt 的总字符预算。直接拿本地窗口预算（单位 token）当字符上限：
# 中文场景下 1 字符约 0.5~1 token，用 token 数当字符上限是保守的（宁可少带）。
# 之所以按「本地窗口」而非「云端窗口」来卡，是因为 tier="cloud" 的请求在云端
# 未接入时会被 llm_client 降级到本地执行——不按本地窗口保护就会超窗。
PROMPT_CHAR_BUDGET = max(1024, local_prompt_budget())

_HEADER = (
    "你是端云协同系统里的汇聚器。下面是针对同一个问题的若干子任务及其答案，"
    "请综合它们，给出面向用户的最终答案。"
)

_REQUIREMENTS = (
    "要求：\n"
    "1. 直接给出面向用户的最终答案，不要罗列推理过程，也不要复述子任务清单。\n"
    "2. 若部分子任务失败或无答案，请基于已有信息作答；信息不足时如实说明，不要编造。\n"
    "3. 用与用户提问相同的语言回答。"
)

# 全失败时的固定说明。放成常量，测试与 CLI 都按同一句话判断，避免各处各写一句。
NO_ANSWER_MESSAGE = "很抱歉，本次所有子任务均未成功完成，无法给出最终答案。请稍后重试或检查模型链路。"

# 降级拼接时的末尾告警。用 {} 占位，避免 f-string 在常量里提前求值。
_FALLBACK_WARNING = "[告警] 汇聚调用失败，以上为各子任务答案的直接拼接，未经整合。原因：{err}"

# 因长度预算被丢弃的那条子任务，用占位行保留其位置——直接删行会让模型以为
# 子任务数量对不上，占位能如实告诉它「这里有一条但没纳入」。
_OMITTED_MARK = "（因上下文长度限制，此条未纳入聚合）"


def aggregate(
    question: str,
    results: list[SubtaskResult],
    *,
    task_id: str = "",
    run_id: str = "default",
) -> str:
    """把子任务结果汇聚成面向用户的最终答案。

    返回的字符串直接作为本次请求的 answer 落库（CLI 负责写 TaskRecord），
    所以这里永远返回一段可展示的文本，绝不抛异常。
    """
    if not results:
        # 没有子任务（例如 planner 退化为不拆且 runner 未执行）视同全失败。
        return NO_ANSWER_MESSAGE

    usable = [r for r in results if r.error is None and r.answer.strip()]
    if not usable:
        # 全失败 / 全无答案：不调 LLM。省一次必然无效的调用，也避免把失败
        # 记成一次成功的 aggregator 调用污染 token 统计。
        return NO_ANSWER_MESSAGE

    prompt = _build_prompt(question, results)
    res = call_llm(
        messages=[{"role": "user", "content": prompt}],
        role="aggregator",
        tier="cloud",
        strategy="dot",
        task_id=task_id,
        run_id=run_id,
        retries=2,
    )

    text = (res.text or "").strip()
    if res.error:
        print(f"[aggregator] ⚠ 汇聚调用失败，降级为答案拼接：{res.error}", file=sys.stderr)
        return _fallback_join(results, res.error)
    if not text:
        # 没报错却返回空文本（端点偶发）：仍按失败处理，否则用户拿到空答案。
        print("[aggregator] ⚠ 汇聚返回空文本，降级为答案拼接", file=sys.stderr)
        return _fallback_join(results, "汇聚返回空文本")
    return text


def _build_prompt(question: str, results: list[SubtaskResult]) -> str:
    """组装聚合 prompt，并在超预算时从**最早**的子任务开始丢（与 runner 一致）。

    为什么从最早丢：后面的子任务通常依赖前面的结论，保留靠后的答案信息密度更高，
    而 DoT 的顺序执行里越早的子任务越可能只是铺垫。
    """
    dropped: set[int] = set()
    while True:
        prompt = _assemble(question, results, dropped)
        if len(prompt) <= PROMPT_CHAR_BUDGET:
            break
        nxt = next((i for i in range(len(results)) if i not in dropped), None)
        if nxt is None:
            # 连占位行都放不下：不再丢，剩给窗口去兜底（这种情况极罕见）。
            break
        dropped.add(nxt)

    if dropped:
        print(
            f"[aggregator] ⚠ 聚合 prompt 超出 {PROMPT_CHAR_BUDGET} 字符预算，"
            f"已省略最早的 {len(dropped)} 条子任务答案",
            file=sys.stderr,
        )
    return prompt


def _assemble(question: str, results: list[SubtaskResult], dropped: set[int]) -> str:
    """把「原始问题 + 子任务→答案清单 + 输出要求」拼成一段 prompt 文本。

    每条子任务单独一行，失败 / 无答案如实标注，让模型自己决定怎么处理——
    隐瞒失败会让模型把空缺当成「没这条」，从而编造缺失的信息。
    """
    blocks = [
        _HEADER,
        "",
        "【原始问题】",
        question.strip(),
        "",
        "【子任务与答案】",
    ]
    for i, r in enumerate(results):
        blocks.append(_render_line(r, i + 1, omitted=(i in dropped)))
    blocks.append("")
    blocks.append(_REQUIREMENTS)
    return "\n".join(blocks)


def _render_line(r: SubtaskResult, label: int, *, omitted: bool) -> str:
    """渲染单条子任务行。omitted 表示因预算被丢，用占位行保住位置。"""
    head = f"{label}. {r.text.strip()}"
    if omitted:
        return f"{head} → {_OMITTED_MARK}"
    if r.error:
        return f"{head} → （执行失败：{_clip(r.error, MAX_ERROR_CHARS)}）"
    ans = r.answer.strip()
    if not ans:
        return f"{head} → （无答案）"
    return f"{head} → {_clip(ans, MAX_ANSWER_CHARS)}"


def _fallback_join(results: list[SubtaskResult], err: object) -> str:
    """汇聚调用失败时的降级：拼接各子任务答案，末尾附一行告警。

    这里用「子任务 → 答案」的原样拼接（不做整合），并带上失败原因，
    让用户/排查者一眼看出这不是模型整合过的结果。
    """
    parts: list[str] = []
    for i, r in enumerate(results):
        head = f"{i + 1}. {r.text.strip()}"
        if r.error:
            parts.append(f"{head} → （执行失败：{_clip(r.error, MAX_ERROR_CHARS)}）")
        elif r.answer.strip():
            parts.append(f"{head} → {r.answer.strip()}")
        else:
            parts.append(f"{head} → （无答案）")
    body = "\n".join(parts)
    return f"{body}\n\n{_FALLBACK_WARNING.format(err=_clip(str(err), MAX_ERROR_CHARS))}"


def _clip(text: str, limit: int) -> str:
    """截断长文本并留痕，避免静默丢信息。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…（已截断）"
