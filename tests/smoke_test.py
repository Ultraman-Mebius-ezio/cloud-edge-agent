"""冒烟测试：不依赖 pytest，直接 ``python -m tests.smoke_test`` 跑。

为什么不用 pytest：本仓库的依赖清单里没有 pytest（见《接口冻结》§2
「不许新增第三方依赖」），而这几条冒烟用例本身也不值得为它引入一个测试框架。
所以这里用最朴素的 assert + 一行 PASS/FAIL + 末尾汇总，够用且零依赖。

为什么每个用例都自己建临时库：冒烟测试绝不能污染 ``runs/metrics.db``。
真实指标库是实验数据的唯一来源，被测试写脏之后分位数会凭空多出几个样本，
而且这种污染不会报错。所以凡是要落库的用例，一律用 ``tempfile`` 建库并把
``db_path`` 显式传给 metrics 的函数（它们都支持这个参数）。

分层原则：纯函数（allocator / planner 解析）与指标往返必须永远能过；
端到端那条要起本地模型、可能很慢，所以放最后并支持 ``--skip-e2e`` 跳过。
本地模型端点不可达时 e2e 判为 SKIP 而不是 FAIL——那是环境问题，不是代码问题。
"""

from __future__ import annotations

import argparse
import inspect
import socket
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

# Windows 控制台默认 GBK，中文会乱码。入口处显式改成 utf-8，
# 与 cli.py 的做法一致（CLAUDE 里要求 Windows 入口都要 reconfigure）。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 迷你测试框架
# ---------------------------------------------------------------------------

class Skip(Exception):
    """环境不满足（比如本地模型没起）。跳过不算失败，但要在汇总里可见。"""


_COUNTS = {"PASS": 0, "FAIL": 0, "SKIP": 0}


def record(name: str, status: str, detail: str = "") -> None:
    _COUNTS[status] += 1
    line = f"[{status}] {name}"
    if detail:
        line += f"  -- {detail}"
    print(line)


def check(name: str, fn) -> None:
    """跑一个用例。fn 正常返回即 PASS，抛断言/异常即 FAIL，抛 Skip 即 SKIP。"""
    try:
        fn()
    except Skip as e:
        record(name, "SKIP", str(e))
    except AssertionError as e:
        record(name, "FAIL", str(e) or "断言失败")
    except Exception as e:  # noqa: BLE001 —— 冒烟层要吞掉一切并如实报告
        record(name, "FAIL", f"{type(e).__name__}: {e}")
    else:
        record(name, "PASS")


# ---------------------------------------------------------------------------
# 1. allocator：纯规则路由（零 LLM 调用）
# ---------------------------------------------------------------------------

# 61 个字长的中性文本：不含任何开放词，只为触发「长度 > 60」这条规则。
# 故意用重复串而不是自然句，避免不小心命中 OPEN_ENDED_WORDS 让用例失真。
LONG_TEXT = "端云协同" * 20  # 80 字


def _load_allocator():
    from src import allocator
    return allocator


def _tier_of(text: str) -> str:
    return _load_allocator().allocate_one(text)


def case_allocator_arith() -> None:
    """纯算术题应被判给本地——它不需要大模型的世界知识。"""
    t = _tier_of("37*48 等于多少")
    assert t == "local", f"期望 local，实际 {t!r}"


def case_allocator_open() -> None:
    """命中「解释/优缺点」等开放词 → 云端。"""
    t = _tier_of("请解释端云协同的优缺点")
    assert t == "cloud", f"期望 cloud，实际 {t!r}"


def case_allocator_long() -> None:
    """超过 60 字 → 云端（长上下文在本地会挤占窗口）。"""
    t = _tier_of(LONG_TEXT)
    assert t == "cloud", f"期望 cloud（长文本 >{60} 字），实际 {t!r}"
    assert len(LONG_TEXT) > 60, "用例自身前提被破坏：LONG_TEXT 不够长"


def case_allocator_table() -> None:
    """allocate 与 allocate_one 必须一致；explain 要等长且非空，供 CLI 展示。"""
    allocator = _load_allocator()
    subtasks = [
        "37*48 等于多少",
        "一个班 45 人 3/5 是女生女生比男生多几人",
        "请解释端云协同的优缺点",
        LONG_TEXT,
    ]
    tiers = allocator.allocate(subtasks)
    assert len(tiers) == len(subtasks), f"allocate 长度 {len(tiers)} != {len(subtasks)}"
    assert all(t in ("local", "cloud") for t in tiers), f"出现非法档位：{tiers}"

    for s, t in zip(subtasks, tiers):
        one = allocator.allocate_one(s)
        assert one == t, f"allocate_one({s!r})={one!r} 与 allocate 的 {t!r} 不一致"

    reasons = allocator.explain(subtasks, tiers)
    assert len(reasons) == len(subtasks), f"explain 长度 {len(reasons)} != {len(subtasks)}"
    assert all(isinstance(r, str) and r for r in reasons), "explain 出现空理由"

    # 打印判定理由表，方便人工核对规则是否按预期命中
    print("    ┌─ allocator 判定理由表 ─────────────────────────────")
    for s, t, r in zip(subtasks, tiers, reasons):
        shown = s if len(s) <= 24 else s[:24] + "…"
        print(f"    │ {shown:<26} {t:<6} {r}")
    print("    └────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# 2. planner：解析容错（把解析函数单独测，不经过 LLM）
# ---------------------------------------------------------------------------

def _get_parse_fn():
    """拿到 planner 的解析函数。

    契约只规定了 decompose()，没规定解析函数叫什么。为了让「解析容错」可被
    单独测试，planner 应把它做成可公开调用的函数；若它是私有实现，这里也
    按私有名去取。两条路都拿不到才报错。
    """
    from src import planner
    for name in ("parse_subtasks", "parse_subtask", "parse", "_parse_subtasks"):
        fn = getattr(planner, name, None)
        if callable(fn):
            return name, fn
    raise AssertionError(
        "planner 未暴露可调用的解析函数（期望 parse_subtasks 或 _parse_subtasks），"
        "无法单独测试解析容错"
    )


def _parse(text: str, question: str) -> list[str]:
    """按解析函数的实际签名调用它，兼容「带 question 兜底参数」与「只吃 text」两种实现。"""
    _, fn = _get_parse_fn()
    params = inspect.signature(fn).parameters
    names = set(params)
    for cand in ("question", "fallback", "original"):
        if cand in names:
            return fn(text, **{cand: question})
    required = [p for p in params.values() if p.default is inspect.Parameter.empty]
    if len(required) >= 2:  # (text, question) 位置参数版
        return fn(text, question)
    return fn(text)


def _norm(s: str) -> str:
    """去掉首尾空白与 markdown 强调符——解析层承诺会做的归一化。"""
    return s.strip().strip("`*# ").strip()


def case_parse_numbered() -> None:
    """规范编号：第一层正则就命中，应逐行抽出、剥掉编号与空白。"""
    text = "1. 计算女生人数\n2. 计算男生人数\n3. 求女生比男生多几人"
    got = [_norm(x) for x in _parse(text, "班上一共有多少人")]
    want = ["计算女生人数", "计算男生人数", "求女生比男生多几人"]
    assert got == want, f"编号输入解析结果不符：{got} != {want}"


def case_parse_plain_lines() -> None:
    """无编号纯换行：第一层抽不到，退到按行切分——结果仍应是合理的多步列表。"""
    text = "先求女生人数\n再求男生人数\n最后算两者之差"
    got = [_norm(x) for x in _parse(text, "班上一共有多少人")]
    want = ["先求女生人数", "再求男生人数", "最后算两者之差"]
    assert got == want, f"纯换行解析结果不符：{got} != {want}"


def case_parse_empty() -> None:
    """空串：三层都拿不到子任务，应退化为「单条」也就是不拆。

    退化是合法结果，不是错误（契约明说）。兜底内容若不依赖原始问题，解析层
    返回空列表也算退化信号——decompose 会在这一层把空结果提升为 [question]。
    所以这里既接受「正好一条」，也接受「空列表」，但不接受多于一条。
    """
    question = "班上一共有多少人"
    got = _parse("   \n  \n", question)
    assert len(got) <= 1, f"空串应退化为至多一条，实际 {len(got)} 条：{got}"
    if got:
        assert _norm(got[0]) == question, (
            f"退化出的单条应为原问题 {question!r}，实际 {got[0]!r}"
        )
        note = f"退化为单条 [question]：{got[0]!r}"
    else:
        note = "解析层返回空列表（由 decompose 提升为 [question]），已视为退化"
    print(f"    └─ 空串退化：{note}")


# ---------------------------------------------------------------------------
# 3. metrics：落库 → 读回 往返
# ---------------------------------------------------------------------------

_CTX: dict = {}


def _prep_metrics() -> dict:
    """写一组已知数值的调用与任务，供两条读回用例比对。"""
    if _CTX:
        return _CTX
    from src import metrics

    tmp = Path(tempfile.mkdtemp(prefix="smoke_metrics_"))
    db = tmp / "metrics.db"
    run_id = "smoke-metrics"
    task_id = "task-" + uuid.uuid4().hex[:8]

    # 三条调用：分解(cloud) + 执行(local) + 执行(cloud)。
    # 云端 token 只在 tier=cloud 时由 CallRecord.finalize 计入 → 150 + 0 + 180 = 330。
    metrics.record(
        metrics.CallRecord(role="decomposer", tier="cloud", model="fake",
                           strategy="dot", run_id=run_id, task_id=task_id,
                           prompt_tokens=100, completion_tokens=50, latency_ms=10.0),
        db_path=db,
    )
    metrics.record(
        metrics.CallRecord(role="executor", tier="local", model="fake",
                           strategy="dot", run_id=run_id, task_id=task_id,
                           prompt_tokens=200, completion_tokens=80, latency_ms=900.0),
        db_path=db,
    )
    metrics.record(
        metrics.CallRecord(role="executor", tier="cloud", requested_tier="cloud", model="fake",
                           strategy="dot", run_id=run_id, task_id=task_id,
                           prompt_tokens=120, completion_tokens=60, latency_ms=20.0),
        db_path=db,
    )
    metrics.record_task(
        metrics.TaskRecord(strategy="dot", run_id=run_id, task_id=task_id,
                           question="Q", answer="A", n_subtasks=2, n_local=1,
                           n_local_planned=1,
                           total_ms=1234.5, cloud_tokens=330, plan_ms=100.0, exec_ms=800.0),
        db_path=db,
    )
    # 第二条请求的子任务数刻意与第一条不同：这是为了钉死 SLM 使用率的口径。
    # 池化口径 Σlocal/Σsub = (1+1)/(2+8) = 0.20；
    # 若误用「每条请求比值的算术平均」会得到 (0.5+0.125)/2 = 0.3125。
    # 只用单请求的测试永远测不出这个分叉（曾经的版本就是这么漏掉的）。
    metrics.record_task(
        metrics.TaskRecord(strategy="dot", run_id=run_id, task_id=task_id + "b",
                           question="Q2", answer="A2", n_subtasks=8, n_local=1,
                           n_local_planned=1,
                           total_ms=200.0, cloud_tokens=0, plan_ms=50.0, exec_ms=150.0),
        db_path=db,
    )
    _CTX.update(db=db, run_id=run_id, task_id=task_id)
    return _CTX


def case_metrics_task_stats() -> None:
    """task_stats 读回请求级数字：n / 云端 token 合计 / SLM 使用率 都要对得上。"""
    ctx = _prep_metrics()
    from src import metrics

    stats = metrics.task_stats(ctx["run_id"], db_path=ctx["db"])
    assert "dot" in stats, f"task_stats 未按 strategy 分组出 dot：{list(stats)}"
    s = stats["dot"]
    assert s["n"] == 2, f"n 期望 2，实际 {s['n']}"
    assert s["cloud_tokens_total"] == 330, f"云端 token 期望 330，实际 {s['cloud_tokens_total']}"
    assert s["p50_ms"] == 717.2, f"p50 期望 717.2((200+1234.5)/2)，实际 {s['p50_ms']}"
    assert s["subtasks_median"] == 5.0, f"子任务中位数期望 5，实际 {s['subtasks_median']}"
    # SLM 使用率必须按口径池化：Σn_local / Σn_subtasks = 2/10，而不是逐请求平均的 0.3125
    assert s["slm_ratio_actual"] == 0.2, (
        f"SLM 使用率(实际) 期望池化 0.2(2/10)，实际 {s['slm_ratio_actual']}；"
        f"若得到 0.3125 说明误用了「逐请求比值的算术平均」"
    )
    assert s["slm_ratio_planned"] == 0.2, (
        f"SLM 使用率(分配) 期望池化 0.2(2/10)，实际 {s['slm_ratio_planned']}"
    )


def case_metrics_role_summary() -> None:
    """role_summary 读回归因：分解 token = 150，执行云端 token = 180。"""
    ctx = _prep_metrics()
    from src import metrics

    rows = metrics.role_summary(ctx["run_id"], db_path=ctx["db"])
    by = {}
    for r in rows:
        by.setdefault(r["role"], {"calls": 0, "cloud": 0})
        by[r["role"]]["calls"] += r["calls"]
        by[r["role"]]["cloud"] += r["cloud_tokens"] or 0

    assert "decomposer" in by, f"role_summary 缺 decomposer：{list(by)}"
    assert by["decomposer"]["calls"] == 1, f"decomposer 调用数期望 1，实际 {by['decomposer']['calls']}"
    assert by["decomposer"]["cloud"] == 150, f"decomposer 云端 token 期望 150，实际 {by['decomposer']['cloud']}"
    assert by["executor"]["calls"] == 2, f"executor 调用数期望 2，实际 {by['executor']['calls']}"
    assert by["executor"]["cloud"] == 180, f"executor 云端 token 期望 180，实际 {by['executor']['cloud']}"

    # 规划开销应能单独拆出来：这里 plan 的云端 token = decomposer(150) + aggregator(0)
    ov = metrics.plan_overhead(ctx["run_id"], db_path=ctx["db"])
    assert ov["plan_cloud_tokens"] == 150, f"规划 token 期望 150，实际 {ov['plan_cloud_tokens']}"
    assert ov["exec_calls"] == 2, f"执行调用数期望 2，实际 {ov['exec_calls']}"


# ---------------------------------------------------------------------------
# 4. 端到端：走 answer_dot（慢，可跳过）
# ---------------------------------------------------------------------------

def _local_endpoint_reachable(timeout: float = 1.0) -> bool:
    """先探一下本地模型端口，避免在模型没起时把超时/重试耗在无用等待上。

    call_llm 配了 request_timeout 与 retries，模型没起时一次请求可能空等很久；
    冒烟测试必须能快速给出结论，所以这里先做个 1 秒的 TCP 探测。
    """
    from src.config import CONFIG
    base = CONFIG.local.api_base or ""
    u = urlparse(base)
    host = u.hostname or "localhost"
    port = u.port or 11434
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def case_e2e_dot() -> None:
    """跑一道多步数学题，验证 answer_dot 全链路能落地。

    三道断言：① 返回非空；② tasks 表里能按 task_id 捞到汇总行；
    ③ 能算出 SLM 使用率（n_subtasks>0 且 n_local 落在 [0, n_subtasks]）。
    """
    from src import config, metrics

    # cli 是集成入口，可能还没写（并行开发期）。缺它属于「未就绪」而不是
    # 测试逻辑本身有缺陷，所以判 SKIP 而不是 FAIL——避免误报。
    try:
        from src import cli  # noqa: PLC0415
    except ImportError as e:
        raise Skip(f"src.cli 尚未就绪，跳过端到端：{e}") from e
    if not hasattr(cli, "answer_dot"):
        raise Skip("src.cli 尚未暴露 answer_dot，跳过端到端")

    if not _local_endpoint_reachable():
        raise Skip(f"本地模型端点不可达（{config.CONFIG.local.api_base}），跳过端到端")

    tmp = Path(tempfile.mkdtemp(prefix="smoke_e2e_"))
    db = tmp / "metrics.db"
    run_id = "smoke-e2e-" + uuid.uuid4().hex[:6]
    task_id = "e2e-" + uuid.uuid4().hex[:8]

    # answer_dot 内部落库走 CONFIG.db_path，这里临时改掉，
    # 保证端到端也不会写进真实的 runs/metrics.db；跑完无论如何都要还原。
    # 走 config.set_db_path() 而不是自己碰 frozen dataclass——那是唯一受支持的入口。
    old_db = config.CONFIG.db_path
    config.set_db_path(db)
    try:
        question = "一个班 45 人，其中 3/5 是女生，女生比男生多几人？"

        params = inspect.signature(cli.answer_dot).parameters
        if "db_path" in params:
            answer, results = cli.answer_dot(question, task_id, run_id, db_path=db)
        else:
            answer, results = cli.answer_dot(question, task_id, run_id)

        assert isinstance(answer, str), f"answer 类型应为 str，实际 {type(answer)}"
        assert answer.strip(), _e2e_failure_hint(results)

        assert results, "answer_dot 未返回子任务结果列表"

        with metrics.session(db) as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        assert row is not None, f"tasks 表里找不到 task_id={task_id}（CLI 是否漏了 record_task？）"

        n_sub = row["n_subtasks"] or 0
        n_local = row["n_local"] or 0
        n_local_planned = row["n_local_planned"] or 0
        assert n_sub > 0, f"n_subtasks 期望 >0，实际 {n_sub}"
        # 这里必须与**实际返回的结果列表**对账，而不是只断言取值区间——
        # `0 <= n_local <= n_sub` 恒真，等于没测（落地时确实这么写过）。
        # 对账才能抓住「落库统计与执行结果不一致」这类接线错误。
        assert n_sub == len(results), (
            f"落库 n_subtasks={n_sub} 与返回的结果数 {len(results)} 不一致"
        )
        expect_local = sum(1 for r in results if r.tier == "local")
        assert n_local == expect_local, (
            f"落库 n_local={n_local}，但结果里实际跑本地的是 {expect_local} 条"
        )
        expect_planned = sum(1 for r in results if r.requested_tier == "local")
        assert n_local_planned == expect_planned, (
            f"落库 n_local_planned={n_local_planned}，"
            f"但分配器判给本地的是 {expect_planned} 条"
        )
        print(
            f"    └─ e2e: 子任务 {n_sub} 个（实际本地 {n_local} / 分配本地 {n_local_planned}）"
            f"  SLM 使用率(分配) {n_local_planned / n_sub:.0%}"
            f"  云端 token {row['cloud_tokens'] or 0}  总耗时 {row['total_ms']:.0f}ms"
        )
    finally:
        config.set_db_path(old_db)


def _e2e_failure_hint(results) -> str:
    """最终答案为空时，把各子任务的错误拼出来，便于判断是环境问题还是逻辑问题。"""
    errs = []
    for r in results or []:
        err = getattr(r, "error", None)
        if err:
            errs.append(f"#{getattr(r, 'index', '?')} {str(err)[:80]}")
    return "最终答案为空；子任务错误：" + ("; ".join(errs) if errs else "无（可疑）")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

CASES = [
    ("allocator: 纯算术 → local", case_allocator_arith),
    ("allocator: 开放问题 → cloud", case_allocator_open),
    ("allocator: 长文本 → cloud", case_allocator_long),
    ("allocator: 逐条一致 + explain 理由表", case_allocator_table),
    ("planner: 规范编号 → 预期列表", case_parse_numbered),
    ("planner: 无编号纯换行 → 合理列表", case_parse_plain_lines),
    ("planner: 空串 → 退化为单条", case_parse_empty),
    ("metrics: task_stats 往返", case_metrics_task_stats),
    ("metrics: role_summary 往返", case_metrics_role_summary),
    ("e2e: answer_dot 多步数学题", case_e2e_dot),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="端云协同系统冒烟测试（无需 pytest）")
    ap.add_argument("--skip-e2e", action="store_true",
                    help="跳过端到端用例（会起本地模型、较慢）")
    args = ap.parse_args(argv)

    print("=" * 66)
    print("冒烟测试 · 端云协同智能体")
    print("=" * 66)
    for name, fn in CASES:
        if args.skip_e2e and name.startswith("e2e"):
            record(name, "SKIP", "命令行指定 --skip-e2e")
            continue
        check(name, fn)

    total = sum(_COUNTS.values())
    print("-" * 66)
    print(f"汇总：{total} 个用例  "
          f"PASS={_COUNTS['PASS']}  FAIL={_COUNTS['FAIL']}  SKIP={_COUNTS['SKIP']}")
    if _COUNTS["FAIL"]:
        print("结果：存在失败用例")
        return 1
    print("结果：全部通过" + ("（有跳过项）" if _COUNTS["SKIP"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
