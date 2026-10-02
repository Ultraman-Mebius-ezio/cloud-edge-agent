"""命令行入口。

本模块只做「编排 + 展示」，三件事刻意不在这里重写：

1. 所有模型调用都经 ``llm_client.call_llm()``（重试与落库都在那里面），
   这里一次 litellm 都不碰——一旦绕过，云端 token 会静默偏低且不报错。
2. 看板数据一律从 metrics 的聚合函数取（task_stats / role_summary /
   plan_overhead），不在 CLI 里另写一套 SQL。否则「口径」会在展示层分叉，
   报告里的数字和库里的数字对不上。
3. 每个请求的云端 token 由数据库按 task_id 汇总（一条 SUM），不在 Python
   里累加——见 ``_sum_cloud_tokens`` 的说明。

为什么入口要自己量 total_ms：申请书口径的「完整响应时间」是**请求级**的
（收到请求 → 最终结果返回，含本地推理）。只有 CLI 站在整个请求的外面，
才量得到这个量纲；模块内部的调用级耗时是另一回事（见 metrics 里的量纲提醒）。
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import allocator, aggregator, config, metrics, planner, runner
from .llm_client import call_llm, warmup

# ---------------------------------------------------------------------------
# 模块级常量：所有可调项集中在这里，不要在函数体里撒魔法数字。
# ---------------------------------------------------------------------------

DEFAULT_MODE = "dot"  # 契约默认主方案
# 执行/汇聚调用的瞬时故障重试次数（契约里 planner/runner/aggregator 均为 2）。
# 本机 sm_120 冷加载偶发 CUDA 初始化失败，重试能吃掉这类抖动。
EXEC_RETRIES = 2
# 交互模式的对话历史滑窗：只保留最近这么多轮（一轮 = 一问一答）。
HISTORY_ROUNDS = 3
# 交互退出词。
EXIT_WORDS = frozenset({"exit", "quit", ":q", ":quit", "退出", "再见"})
# 对话系统提示词：仅 direct 交互使用（dot 走分解流水线，不需要它）。
CHAT_SYSTEM = (
    "你是端云协同智能助手。请直接、简洁地回答用户，"
    "不要复述问题，不要输出多余的客套话。"
)
# cloud_ready() 为假时，看板末尾必须亮出的降级提示。
CLOUD_DEGRADED_NOTICE = (
    "云端未接入，本表数字为本地降级运行，不代表真实对照结论。"
    "（tier=cloud 的调用实际在本地执行，requested_tier 才记录原请求档位）"
)

console = Console()


# ---------------------------------------------------------------------------
# 入口与参数
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 主入口。返回进程退出码（0 正常）。"""
    _setup_std_streams()
    args = _parse_args(argv)

    # --db 必须尽早生效，否则 call_llm 已经把行写进默认库了。
    if args.db:
        _override_db(args.db)

    # 看板只读库，跳过预热与 run_id 生成（没有 --run-id 就统计全部批次）。
    if args.stats:
        _print_stats(args.run_id)
        return 0

    run_id = args.run_id or _auto_run_id()

    if not args.no_warmup:
        _do_warmup()

    if args.once is not None:
        return _run_once(args.once, args.mode, run_id)

    return _interactive(args.mode, run_id)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="端云协同高效智能体：direct=整请求规则路由（对照），dot=子任务级分配（主方案）。",
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "dot"),
        default=DEFAULT_MODE,
        help="direct = 整请求路由（对照基线）；dot = 子任务级分配（主方案，默认）",
    )
    parser.add_argument(
        "--once",
        metavar="问题",
        default=None,
        help="单次问答后退出（不做交互）",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="只打印指标看板后退出",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="实验批次号；不指定则自动生成，如 run-20261002-2130",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="跳过启动预热（默认会预热本地模型）",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="覆盖 metrics.db 路径",
    )
    return parser.parse_args(argv)


def _setup_std_streams() -> None:
    """把 stdout/stderr 切到 UTF-8。

    Windows 的 GBK 控制台会把中文打成乱码（reconfigure 之前也会在打印
    非 GBK 字符时抛 UnicodeEncodeError）。errors="replace" 保证极端情况下
    宁可显示成问号也不要崩掉。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # 某些被重定向的流没有 reconfigure
            pass


def _override_db(db: str) -> None:
    """让 ``--db`` 对整条链路同时生效。

    必须改单例而不是「只对自己这一侧传 db_path」：call_llm 落库走的是
    metrics.record -> session(CONFIG.db_path)，若不同步，calls 表会留在默认库、
    tasks 表落到新库，按 task_id 汇总云端 token 将恒为 0——静默失真，最难发现。
    改单例的口子收在 config.set_db_path() 里，这里不自己动 frozen dataclass。
    """
    path = config.set_db_path(db)
    console.print(f"[dim]指标库：{path}[/]")


def _auto_run_id() -> str:
    """自动生成实验批次号，形如 run-20261002-2130（本地时间）。"""
    return "run-" + time.strftime("%Y%m%d-%H%M")


def _new_task_id() -> str:
    """一次请求的主键：请求内所有调用共享它，事后按它把整条链路捞出来。"""
    return uuid.uuid4().hex[:12]


def _do_warmup() -> None:
    """预热本地模型，把冷加载耗时从正式测量里剔除（不预热 P95 完全失真）。"""
    console.print("[dim]预热本地模型（冷加载偶发 CUDA 失败，warmup 内部自带重试）…[/]")
    dt = warmup("local")
    console.print(f"[dim]预热完成：{dt:.2f}s[/]")


# ---------------------------------------------------------------------------
# 两种回答模式（契约 §2 cli）
# ---------------------------------------------------------------------------


def answer_direct(question: str, task_id: str, run_id: str) -> tuple[str, dict[str, Any]]:
    """direct 对照臂：整请求一次性路由到某个档位后答完。

    ⚠ 这里的 tier 来自 ``allocator.allocate_one`` 的**规则**判定（长度 / 关键词 /
    算术），它没有任何学习成分，**不是训练出来的路由器**。报告里只能把它当作
    「规则整请求路由」的对照臂，**绝不能当强基线**，更不能用它的数字去证明
    「学习型路由」的收益——DoT 论文里不训练时 P3 只有 15% 量级，把这种桩当强
    对手会把结论带偏。
    """
    return _direct_impl(question, task_id, run_id, stream=False, on_delta=None, history=None)


def _direct_impl(
    question: str,
    task_id: str,
    run_id: str,
    *,
    stream: bool,
    on_delta: Callable[[str], None] | None,
    history: Sequence[tuple[str, str]] | None,
) -> tuple[str, dict[str, Any]]:
    """direct 的实际实现。契约签名只暴露 3 个参数，流式与历史仅交互模式内部用。"""
    t0 = time.perf_counter()
    tier = allocator.allocate_one(question)  # ← 规则路由，非学习路由器（见 answer_direct 注释）
    messages = _build_messages(question, history)
    result = call_llm(
        messages=messages,
        role="executor",
        tier=tier,
        strategy="direct",
        task_id=task_id,
        run_id=run_id,
        retries=EXEC_RETRIES,
        stream=stream,
        on_delta=on_delta,
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    answer = result.text

    _record_task(
        strategy="direct",
        task_id=task_id,
        run_id=run_id,
        question=question,
        answer=answer,
        results=None,  # direct 无子任务 → n_subtasks / n_local 记 0
        total_ms=total_ms,
        plan_ms=0.0,
        exec_ms=result.latency_ms,
        error=result.error,
    )
    meta: dict[str, Any] = {
        "tier": result.tier,
        "requested_tier": result.requested_tier,
        "fell_back": result.fell_back,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_ms": result.latency_ms,
        "usage_missing": result.usage_missing,
        "error": result.error,
    }
    return answer, meta


def answer_dot(
    question: str, task_id: str, run_id: str
) -> tuple[str, list[runner.SubtaskResult]]:
    """dot 主方案：分解 → 分配 → 顺序执行 → 汇聚。"""
    t0 = time.perf_counter()

    # 规划阶段（分解 + 稍后的汇聚）的耗时单独累计，用于「规划开销值不值」的对照。
    t_plan = time.perf_counter()
    subtasks = planner.decompose(question, task_id=task_id, run_id=run_id)
    tiers = allocator.allocate(subtasks)
    plan_ms = (time.perf_counter() - t_plan) * 1000.0

    results = runner.run(
        subtasks, tiers, question=question, task_id=task_id, run_id=run_id
    )
    exec_ms = sum(r.latency_ms for r in results)

    t_agg = time.perf_counter()
    answer = aggregator.aggregate(question, results, task_id=task_id, run_id=run_id)
    plan_ms += (time.perf_counter() - t_agg) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0

    _record_task(
        strategy="dot",
        task_id=task_id,
        run_id=run_id,
        question=question,
        answer=answer,
        results=results,
        total_ms=total_ms,
        plan_ms=plan_ms,
        exec_ms=exec_ms,
        error=_task_error(results),
    )
    return answer, results


def _task_error(results: list[runner.SubtaskResult]) -> str | None:
    """把「整条流水线失败」汇总成请求级 error。

    必须和 direct 臂同口径：`metrics.task_stats` 用 `error IS NULL` 过滤样本，
    direct 写 error 而 dot 不写，就会造成 direct 的失败请求被剔除、
    dot 的失败请求被当成正常样本计入 P50/P95 与 SLM 使用率——
    两臂数字不可比，且完全静默。这是本项目最忌讳的那类失真。

    判据与 aggregator 一致：只要还有一条子任务拿到了非空答案，就算这次请求
    有可用产出（汇聚即便降级拼接也给了东西），不记 error。
    """
    usable = [r for r in results if r.error is None and r.answer.strip()]
    if usable:
        return None
    if not results:
        return "planner 未产出子任务"
    first_err = next((r.error for r in results if r.error), None)
    return f"全部 {len(results)} 个子任务均失败：{first_err or '答案为空'}"


def _build_messages(
    question: str, history: Sequence[tuple[str, str]] | None
) -> list[dict[str, str]]:
    """把滑窗历史拼成 OpenAI 格式消息。history 为空时就是单轮问答。"""
    messages: list[dict[str, str]] = [{"role": "system", "content": CHAT_SYSTEM}]
    for prev_q, prev_a in history or ():
        messages.append({"role": "user", "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a})
    messages.append({"role": "user", "content": question})
    return messages


# ---------------------------------------------------------------------------
# 落库：请求级汇总
# ---------------------------------------------------------------------------


def _record_task(
    *,
    strategy: str,
    task_id: str,
    run_id: str,
    question: str,
    answer: str,
    results: list[runner.SubtaskResult] | None,
    total_ms: float,
    plan_ms: float,
    exec_ms: float,
    error: str | None = None,
) -> None:
    """写一行 tasks。cloud_tokens 来自数据库汇总，不在这里累加。

    本地子任务数记**两个**：n_local 是实际在哪跑的，n_local_planned 是分配器
    判定的。两者平时相同，但云端不可用时会分叉——此时实际全是 local，
    只报 n_local 会让 SLM 使用率恒为 100%，把分配器的判断完全掩盖掉，
    而分配器的判断才是要研究的东西。
    """
    n_subtasks = len(results) if results else 0
    n_local = sum(1 for r in results if r.tier == "local") if results else 0
    n_local_planned = (
        sum(1 for r in results if r.requested_tier == "local") if results else 0
    )
    metrics.record_task(
        metrics.TaskRecord(
            strategy=strategy,
            task_id=task_id,
            run_id=run_id,
            question=question,
            answer=answer,
            n_subtasks=n_subtasks,
            n_local=n_local,
            n_local_planned=n_local_planned,
            total_ms=round(total_ms, 1),
            cloud_tokens=_sum_cloud_tokens(task_id),
            plan_ms=round(plan_ms, 1),
            exec_ms=round(exec_ms, 1),
            error=error,
        )
    )


def _sum_cloud_tokens(task_id: str) -> int:
    """按 task_id 从库里汇总云端 token。

    为什么不在 Python 里累加：一次 dot 请求的调用点有 k+2 个，且重试与失败尝试
    会各自写一行（申请书口径要求失败调用的 token 也算）。手工累加极易漏掉某一行的
    失败尝试——而这类漏算不会报错，只会让云端 token 静默偏低，等发现时实验要重跑。
    交给数据库一条 SQL 算完，主键 task_id 保证一行不漏。
    """
    if not task_id:
        return 0
    try:
        with metrics.session() as conn:
            row = conn.execute(
                "SELECT SUM(cloud_tokens) FROM calls WHERE task_id = ?", (task_id,)
            ).fetchone()
        return int(row[0] or 0) if row else 0
    except Exception as e:  # 汇总失败不能打断主流程，但要吵
        print(f"[cli] 汇总云端 token 失败：{e!r}", file=sys.stderr)
        return 0


# ---------------------------------------------------------------------------
# 单次模式
# ---------------------------------------------------------------------------


def _run_once(question: str, mode: str, run_id: str) -> int:
    task_id = _new_task_id()
    console.print(f"[dim]run_id={escape(run_id)}  task_id={task_id}  mode={mode}[/]")

    if mode == "direct":
        answer, meta = answer_direct(question, task_id, run_id)
        _print_answer(answer)
        _print_direct_meta(meta)
    else:
        answer, results = answer_dot(question, task_id, run_id)
        _print_dot_plan(results)
        _print_answer(answer)
    return 0


# ---------------------------------------------------------------------------
# 交互模式
# ---------------------------------------------------------------------------


def _interactive(mode: str, run_id: str) -> int:
    """多轮对话。历史用最简单滑窗（最近 HISTORY_ROUNDS 轮，常量可改）。

    流式：direct 模式下单次调用可流式，逐 token 打印；
    dot 模式下最终答案来自 ``aggregator.aggregate``（契约冻结为非流式），
    且子任务执行也被 runner 冻结为非流式，所以 dot 交互不做逐字流式输出，
    而是等汇聚完成后整段展示。token 一律以 usage 为准，不数显示的字符。
    """
    _print_banner(mode, run_id)
    history: list[tuple[str, str]] = []

    while True:
        try:
            raw = console.input("[bold cyan]你>[/] ")
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / 提示符处 Ctrl+C：干净退出，不打 traceback
            console.print()
            _print_goodbye()
            return 0

        question = raw.strip()
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            _print_goodbye()
            return 0

        task_id = _new_task_id()
        try:
            if mode == "direct":
                console.print(Rule("direct · 规则整请求路由"))
                answer, meta = _direct_impl(
                    question,
                    task_id,
                    run_id,
                    stream=True,
                    on_delta=_make_delta_printer(),
                    history=history[-HISTORY_ROUNDS:],
                )
                console.print()  # 结束流式那一行
                _print_direct_meta(meta)
            else:
                answer, results = answer_dot(question, task_id, run_id)
                _print_dot_plan(results)
                _print_answer(answer)
        except KeyboardInterrupt:
            # 回答过程中 Ctrl+C：中断本次请求，回到提示符，不打 traceback
            console.print("\n[yellow]已中断本次请求。[/]")
            continue

        history.append((question, answer))
        del history[:-HISTORY_ROUNDS]  # 滑窗：只留最近 N 轮

    return 0  # pragma: no cover - 循环内 return


def _make_delta_printer() -> Callable[[str], None]:
    """流式增量打印器。markup/highlight 关掉，避免模型输出里的 [ ] 被 rich 当样式。"""

    def on_delta(piece: str) -> None:
        console.print(piece, end="", markup=False, highlight=False, soft_wrap=True)

    return on_delta


# ---------------------------------------------------------------------------
# 展示层
# ---------------------------------------------------------------------------


def _print_banner(mode: str, run_id: str) -> None:
    console.print(Rule("端云协同智能体"))
    if config.cloud_ready():
        cloud_txt = "[green]已接入[/]"
    else:
        cloud_txt = "[yellow]未接入（cloud 调用将降级本地）[/]"
    console.print(
        f"mode=[bold]{mode}[/]   run_id={escape(run_id)}   云端={cloud_txt}"
    )
    console.print("[dim]输入问题回车提问；exit 或 Ctrl+C 退出。[/]")


def _print_goodbye() -> None:
    console.print("[dim]再见。[/]")


def _print_dot_plan(results: list[runner.SubtaskResult]) -> None:
    """打印分解列表 + 每个子任务的档位（验收②要求可见）。"""
    if not results:
        return
    console.print(Rule(f"分解为 {len(results)} 个子任务"))
    for i, r in enumerate(results, 1):
        tag = r.tier
        if r.requested_tier and r.requested_tier != r.tier:
            tag = f"{r.requested_tier}→{r.tier} 降级"
        line = Text()
        line.append(f"  {i}. ", style="bold")
        line.append(f"[{tag}] ", style="cyan")
        line.append(r.text)
        if r.dropped_context:
            line.append("  (前置上下文已截断)", style="yellow")
        if r.error:
            line.append(f"  ✗ {r.error[:80]}", style="red")
        console.print(line)


def _print_answer(answer: str) -> None:
    console.print(Rule("最终答案"))
    console.print(Panel(Text(answer or "(空)"), border_style="green", expand=False))


def _print_direct_meta(meta: dict[str, Any]) -> None:
    bits = [f"tier={meta.get('tier')}"]
    if meta.get("fell_back"):
        bits.append(f"请求={meta.get('requested_tier')}（已降级到本地）")
    bits.append(
        f"tokens={meta.get('prompt_tokens')}+{meta.get('completion_tokens')}"
    )
    bits.append(f"latency={float(meta.get('latency_ms') or 0):.0f}ms")
    if meta.get("usage_missing"):
        bits.append("⚠ 未返回 usage，token 记 0")
    if meta.get("error"):
        bits.append(f"error={meta['error']}")
    console.print("[dim]" + escape("  ".join(bits)) + "[/]")


# ---------------------------------------------------------------------------
# --stats 看板
# ---------------------------------------------------------------------------


def _print_stats(run_id: str | None) -> None:
    """三张表 + 降级提示。数据全部来自 metrics 的聚合函数，CLI 不现写 SQL。"""
    console.print(Rule("指标看板"))
    console.print(f"[dim]run_id = {escape(run_id) if run_id else '全部批次'}[/]")

    _print_task_table(run_id)
    _print_plan_table(run_id)
    _print_role_table(run_id)

    if not config.cloud_ready():
        console.print()
        console.print(
            Panel(Text(CLOUD_DEGRADED_NOTICE), border_style="yellow", title="注意")
        )


def _print_task_table(run_id: str | None) -> None:
    """表1：请求级对照。分位数必须来自 tasks 表（量纲见 metrics 的提醒）。"""
    stats = metrics.task_stats(run_id)
    table = Table(
        title="表1 请求级对照（完整响应时间，量纲为请求）", header_style="bold"
    )
    table.add_column("strategy")
    table.add_column("n", justify="right")
    table.add_column("p50(ms)", justify="right")
    table.add_column("p95(ms)", justify="right")
    table.add_column("云端token合计", justify="right")
    table.add_column("SLM使用率(分配)", justify="right")
    table.add_column("SLM使用率(实际)", justify="right")
    table.add_column("子任务中位数", justify="right")

    if not stats:
        table.add_row(*(["—"] * 8))
    for strategy in sorted(stats):
        s = stats[strategy]
        planned = s["slm_ratio_planned"]
        actual = s["slm_ratio_actual"]
        table.add_row(
            strategy,
            str(s["n"]),
            f"{s['p50_ms']:.1f}",
            f"{s['p95_ms']:.1f}",
            str(s["cloud_tokens_total"]),
            "—" if planned is None else f"{planned * 100:.1f}%",
            "—" if actual is None else f"{actual * 100:.1f}%",
            f"{s['subtasks_median']:.1f}",
        )
    console.print(table)
    if any(
        (s["slm_ratio_planned"] is not None and s["slm_ratio_actual"] is not None
         and s["slm_ratio_planned"] != s["slm_ratio_actual"])
        for s in stats.values()
    ):
        console.print(
            "[yellow]两个 SLM 使用率不一致：分配器判定与实际执行分叉，"
            "说明有子任务被降级。**报告里引用「分配」那一列**——"
            "「实际」列在云端不可用时恒为 100%，不反映分配策略。[/]"
        )


def _print_plan_table(run_id: str | None) -> None:
    """表2：规划开销（分解+汇聚）与执行开销分开，回答「拆解本身值不值」。

    只统计 dot：规划这两个角色只存在于 dot 流水线，而同一个 run_id 下通常
    还跑过 direct。不锁策略的话，direct 那一发 executor 会被算进「执行开销」——
    数字看着合理，口径已经错了（plan_overhead 默认 strategy="dot"）。
    """
    po = metrics.plan_overhead(run_id, strategy="dot")
    table = Table(title="表2 规划开销 vs 执行开销（仅 dot）", header_style="bold")
    table.add_column("项目")
    table.add_column("云端token", justify="right")
    table.add_column("调用次数", justify="right")
    table.add_column("耗时合计(ms)", justify="right")
    table.add_row(
        "规划 分解+汇聚",
        str(po["plan_cloud_tokens"]),
        str(po["plan_calls"]),
        f"{po['plan_ms_total']:.1f}",
    )
    table.add_row(
        "执行 子任务",
        str(po["exec_cloud_tokens"]),
        str(po["exec_calls"]),
        f"{po['exec_ms_total']:.1f}",
    )
    console.print(table)
    if po["plan_calls"] and po["plan_cloud_tokens"] == 0:
        console.print(
            "[yellow]规划调用次数非零但云端 token 为 0：云端未接入，"
            "规划实际在本地降级运行。[/]"
        )


def _print_role_table(run_id: str | None) -> None:
    """表3：role × tier 明细，含降级次数（requested_tier != tier 的行数）。"""
    rows = metrics.role_summary(run_id)
    table = Table(title="表3 role × tier 调用明细", header_style="bold")
    table.add_column("role")
    table.add_column("tier")
    for col in ("调用", "云端token", "总token", "耗时合计(ms)", "降级次数", "失败次数"):
        table.add_column(col, justify="right")

    if not rows:
        table.add_row(*(["—"] * 8))
    for r in rows:
        table.add_row(
            r["role"],
            r["tier"],
            str(r["calls"]),
            str(r["cloud_tokens"] or 0),
            str(r["all_tokens"] or 0),
            f"{r['latency_ms_total'] or 0:.1f}",
            str(r["fallback_calls"] or 0),
            str(r["failed_calls"] or 0),
        )
    console.print(table)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
