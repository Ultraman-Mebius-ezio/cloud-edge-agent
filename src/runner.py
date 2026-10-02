"""子任务执行器：顺序执行，把已完成的答案喂给下一个。

为什么单独抽这一层，而不是把逻辑塞进 CLI：
- 三指标里的「完整响应时间」与「云端 token」都在执行阶段逐次累加，执行阶段的
  边界必须清晰可测（谁调用、带什么上下文、花了多少 token）。CLI 拿不到中间态。
- 预算控制只有在能看到「历次调用真实回报的 token」时才成立，而这份账只有执行器
  自己持有。

两条硬约束（来自 docs/接口冻结.md 第 0 节）：
1. 每一次执行都必须走 llm_client.call_llm()，不许绕过——绕过会让云端 token
   静默偏低，而且不报错，等发现时实验全部要重跑。
2. tier 记**实际**执行档位（云端降级后是 local），requested_tier 记分配器给的
   档位；两者都要保留，报告里才能算降级率。

关于顺序执行与「全量前缀」：v4 明确不做依赖图、不并行。第 i 个的 prompt 带上
0..i-1 **全部**已完成答案，这是对 DoT 原文「只带直接前置依赖」的简化（报告里
要写明差别）。代价是 prompt 随 i 线性变长，所以本地窗口的预算控制不是可选项。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from . import llm_client
from .config import CONFIG, cloud_ready, local_prompt_budget

# —— 模块级常量：调用参数与提示词骨架集中在此，避免魔法值散落 ——

ROLE = "executor"          # 必须在 llm_client.ROLES 里（已登记）
STRATEGY = "dot"
DEFAULT_RETRIES = 2        # 本机 sm_120 冷加载偶发 CUDA 初始化失败，靠重试吃掉

# 子任务 prompt 结构。骨架照抄接口冻结文档 §2，改动会同时影响 planner 的
# 提示词设计与预算核算（固定开销是按这份模板实测出来的）。
PROMPT_HEADER = "你是端云协同系统里的执行器。请只回答下面这一个子任务，不要复述整个问题。"
SECTION_ORIGINAL = "【原始问题】"
SECTION_PRIOR = "【此前已完成的子任务与答案】"
SECTION_CURRENT = "【当前子任务】"
ANSWER_ARROW = " → "


@dataclass
class SubtaskResult:
    """一个子任务的一次执行结果。

    为什么要同时留 tier 与 requested_tier：云端不可用时 call_llm 会把
    tier=cloud 的调用**真的**降级到本地模型跑。若只记一个档位，报告里就分不清
    「这次是本地模型发力」还是「云端被降级」，降级率也算不出来。两者之差
    （等价于 LLMResult.fell_back）就是降级的事实。
    """

    index: int
    text: str
    tier: str            # 实际执行档位（降级后为 "local"）
    requested_tier: str  # allocator 分配的档位
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    dropped_context: bool = False  # 本次是否因预算触发了前置答案截断
    error: str | None = None


# —— 进度/告警输出 ——
# 约定：stdout 只留给最终答案（CLI 可能被管道或重定向消费），进度与告警一律
# 走 stderr，免得混进结果里。TTY 下用 rich 上色，非 TTY 退回朴素 print。
try:
    from rich.console import Console as _RichConsole

    _CONSOLE = _RichConsole(stderr=True)
except Exception:  # pragma: no cover - rich 本就在依赖里，兜底只为不阻塞主流程
    _CONSOLE = None


def _emit(msg: str) -> None:
    if _CONSOLE is not None and _CONSOLE.is_terminal:
        # markup=False 很关键：进度行形如 "[3/5] ..."，rich 默认会把它当 markup 标签，
        # 轻则吃掉方括号，重则抛 MarkupError。
        _CONSOLE.print(msg, highlight=False, markup=False)
    else:
        print(msg, file=sys.stderr, flush=True)


def _predicted_tier(requested: str) -> str:
    """不实际调用，先预判本次会跑在哪个档位。

    预算必须在**构造 prompt 之前**判定，此时还没拿到 LLMResult，所以这里按
    call_llm 的降级规则复刻一次判定（见 llm_client.call_llm 的 ① 段）：
    云端不可用且开了降级开关时，请求 cloud 的调用实际会落在本地，本地窗口是
    硬约束，因此同样要纳入预算。
    """
    if requested == "cloud" and not cloud_ready() and CONFIG.cloud_fallback_local:
        return "local"
    return requested


def _build_prompt(question: str, included: list[SubtaskResult], current: str) -> list[dict]:
    """拼装子任务 prompt（结构见接口冻结文档 §2）。"""
    parts = [PROMPT_HEADER, "", SECTION_ORIGINAL + question, ""]
    if included:
        parts.append(SECTION_PRIOR)
        for n, r in enumerate(included, 1):
            parts.append(f"{n}. {r.text}{ANSWER_ARROW}{r.answer}")
        parts.append("")
    parts.append(SECTION_CURRENT + current)
    return [{"role": "user", "content": "\n".join(parts)}]


def _context_tokens(included: list[SubtaskResult]) -> int:
    """前置答案占用的 prompt token，取历次调用真实回报的值。

    每个答案的实际 token 数就是产出它的那次调用的 completion_tokens——这是
    「精确预算」的全部依据。**绝不用 len(text) 估**：那正是接口冻结文档第 0 节
    第 3 条红线禁止的做法，猜出来的数会让截断时机出错且难以察觉。
    """
    return sum(r.completion_tokens for r in included)


def _drop_to_fit(
    completed: list[SubtaskResult], fixed_overhead: int, budget: int
) -> list[int]:
    """就地丢弃最早的若干前置答案，直到「固定开销 + 剩余答案」放进预算。

    返回被丢弃的子任务序号，供告警使用。淘汰是单调的：一旦丢掉最早的，后续调用
    只会丢得更多、不会又捡回来，因此 dropped_context 标记不会被反复推翻。
    """
    dropped: list[int] = []
    while completed and fixed_overhead + _context_tokens(completed) > budget:
        victim = completed.pop(0)  # 从最早的开始丢
        victim.dropped_context = True
        dropped.append(victim.index)
    return dropped


def run(
    subtasks: list[str],
    tiers: list[str],
    *,
    question: str,
    task_id: str = "",
    run_id: str = "default",
) -> list[SubtaskResult]:
    """顺序执行全部子任务，返回与 subtasks 等长的结果列表。

    - 单个子任务失败**不中断**整体：该条 answer=""、error 记下，继续跑后面的。
      一次请求里可能有 k 个子任务，任何一个炸掉就整条请求作废，会让 P95 被
      「本来能跑完的长请求」污染。
    - 预算只对**本地**调用生效（本地窗口是硬约束）；cloud 始终带全量前置答案。
    """
    subs = list(subtasks)
    tier_list = list(tiers)
    if len(tier_list) < len(subs):
        # 分配器若少给了档位，保守补 "local"（本地更受限，宁可多走预算）。
        tier_list.extend(["local"] * (len(subs) - len(tier_list)))

    results: list[SubtaskResult] = []
    # 固定开销（问题本身 + 模板 + 一个子任务）：用**第一次成功调用**回报的
    # prompt_tokens 减去当时前置答案 token 得到，之后整条请求复用这个常数。
    # 首次调用前置答案为空，所以它约等于「问题 + 模板 + 子任务文本」。用它做
    # 常数会略微高估（把参考子任务的文本也算进固定开销），偏保守、安全。
    fixed_overhead: int | None = None
    budget = local_prompt_budget()
    total = len(subs)

    for i, text in enumerate(subs):
        requested = tier_list[i]
        predicted = _predicted_tier(requested)

        # 只有「已完成且有答案」的前置结果进上下文；失败的没有答案，不喂给模型，
        # 也不占预算。
        completed = [r for r in results if r.answer and r.error is None]

        # 预算控制（必须在建 prompt 之前，因为要决定带哪些答案）。
        # 仅本地窗口是硬约束；请求 cloud 且未被降级时带全量前置答案。
        if predicted == "local" and fixed_overhead is not None:
            dropped = _drop_to_fit(completed, fixed_overhead, budget)
            if dropped:
                _emit(
                    f"[runner] ⚠ 本地预算 {budget} 放不下全部前置答案，"
                    f"已丢弃最早 {len(dropped)} 条（子任务序号 {dropped}）"
                )

        messages = _build_prompt(question, completed, text)
        # 记录本次 prompt 实际带进去的前置答案 token，用于反推固定开销。
        ctx_tokens = _context_tokens(completed)

        _emit(f"[{i + 1}/{total}] tier={predicted} 正在执行…")

        t0 = time.perf_counter()
        res = llm_client.call_llm(
            messages=messages,
            role=ROLE,
            tier=requested,
            strategy=STRATEGY,
            task_id=task_id,
            run_id=run_id,
            retries=DEFAULT_RETRIES,
        )

        # usage 缺失会让三指标失真，必须让它可见（接口冻结文档第 0 节第 3 条）。
        if res.error is None and res.usage_missing:
            _emit(
                f"[runner] ⚠ 子任务 {i} 未返回 usage，token 记为 0，指标不可信"
            )

        # 首个成功调用的 prompt_tokens 减去当时的前置答案 token = 固定开销常数。
        # 只在**实际跑在本地**的调用上取：这个常数是给本地窗口预算用的，
        # 若拿一次云端调用的 prompt_tokens 来定，就带进了云端分词器的口径，
        # 两边 token 定义不同，预算会系统性偏。
        if (
            fixed_overhead is None
            and res.error is None
            and res.prompt_tokens > 0
            and res.tier == "local"
        ):
            fixed_overhead = max(0, res.prompt_tokens - ctx_tokens)

        sr = SubtaskResult(
            index=i,
            text=text,
            # tier 用 llm_client 回的**实际**档位；若为空再退回预判值。
            tier=res.tier or predicted,
            requested_tier=res.requested_tier or requested,
            answer=res.text if res.error is None else "",
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            latency_ms=res.latency_ms,
            error=res.error,
        )
        results.append(sr)

        # 完成后追加耗时与 token。耗时统一从 latency_ms 取（客户端计时为准）。
        elapsed = sr.latency_ms / 1000.0
        if sr.error is None:
            _emit(
                f"[{i + 1}/{total}] tier={sr.tier} 完成 {elapsed:.1f}s "
                f"tok {sr.prompt_tokens}+{sr.completion_tokens}"
            )
        else:
            _emit(f"[{i + 1}/{total}] tier={sr.tier} 失败 {elapsed:.1f}s: {sr.error}")

    return results
