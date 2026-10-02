"""三指标落库。

设计要点：
- 每一次 LLM 调用都无条件写一行（含失败的调用），否则"云端 token 累计
  所有 API 调用"这一口径会被静默打破，而且不报错。
- 用 WAL 模式 + 短连接，避免并发写入锁死。
- 客户端计时为准：网关返回的 usage 用于交叉校验，时间用本地 perf_counter。
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  call_id           TEXT PRIMARY KEY,
  run_id            TEXT,
  task_id           TEXT,
  ts                TEXT,
  strategy          TEXT,      -- dot | direct | warmup | smoke
  role              TEXT,      -- decomposer | executor | aggregator | warmup
  tier              TEXT,      -- local | cloud（已降级后的**实际**执行档位）
  requested_tier    TEXT,      -- 调用方原本要求的档位；与 tier 不同即为降级
  model             TEXT,
  is_cache_hit      INTEGER DEFAULT 0,
  cache_similarity  REAL,
  prompt_tokens     INTEGER DEFAULT 0,
  completion_tokens INTEGER DEFAULT 0,
  cache_hit_tokens  INTEGER DEFAULT 0,
  latency_ms        REAL,
  ttft_ms           REAL,
  cloud_tokens      INTEGER DEFAULT 0,   -- 仅 tier=cloud 时非零
  correct           INTEGER,
  error             TEXT,
  raw_usage         TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_run ON calls(run_id);
CREATE INDEX IF NOT EXISTS idx_calls_task ON calls(task_id);

-- 一次用户请求 = 一行。三指标里的「完整响应时间」是请求级量纲，
-- 落在 calls 上算分位数会把「一次请求拆成 5 次调用」的系统算成 5 倍样本，
-- 分位数直接失真。所以必须单独一张表。
CREATE TABLE IF NOT EXISTS tasks (
  task_id       TEXT PRIMARY KEY,
  run_id        TEXT,
  ts            TEXT,
  strategy      TEXT,      -- dot | direct
  question      TEXT,
  answer        TEXT,
  n_subtasks    INTEGER DEFAULT 0,
  n_local       INTEGER DEFAULT 0,   -- 实际在本地跑的
  n_local_planned INTEGER DEFAULT 0, -- 分配器判定给本地的（降级时会与 n_local 分叉）
  total_ms      REAL,      -- 收到请求 → 最终结果返回（含本地推理）
  cloud_tokens  INTEGER DEFAULT 0,
  plan_ms       REAL,      -- 分解 + 汇聚的耗时
  exec_ms       REAL,      -- 子任务执行耗时
  error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);
"""


@dataclass
class CallRecord:
    role: str
    tier: str
    model: str
    strategy: str = "dot"
    requested_tier: str = ""
    run_id: str = "default"
    task_id: str = ""
    is_cache_hit: bool = False
    cache_similarity: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    cloud_tokens: int = 0
    correct: int | None = None
    error: str | None = None
    raw_usage: str | None = None
    call_id: str = ""
    ts: str = ""

    def finalize(self) -> "CallRecord":
        if not self.call_id:
            self.call_id = uuid.uuid4().hex
        if not self.ts:
            self.ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        # 云端 token 是核心指标：只有 tier=cloud 才计入。
        # 本地 token 不计入消耗，但本地推理耗时计入响应时间。
        self.cloud_tokens = (
            (self.prompt_tokens or 0) + (self.completion_tokens or 0)
            if self.tier == "cloud"
            else 0
        )
        return self


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@contextmanager
def session(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    from .config import CONFIG

    conn = _connect(db_path or CONFIG.db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


# 早期版本的库缺列。CREATE TABLE IF NOT EXISTS 不会补列，所以这里显式补，
# 免得开发中途改了 schema 之后，老库一 INSERT 就炸 "no such column"。
_MIGRATIONS = {
    "calls": {"requested_tier": "TEXT"},
    "tasks": {"n_local_planned": "INTEGER DEFAULT 0"},
}


def _migrate(conn: sqlite3.Connection) -> None:
    for table, columns in _MIGRATIONS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def record(rec: CallRecord, db_path: Path | None = None) -> CallRecord:
    """把一次调用写入库。任何异常都不应打断主流程。"""
    rec.finalize()
    row = asdict(rec)
    cols = [
        "call_id", "run_id", "task_id", "ts", "strategy", "role", "tier",
        "requested_tier", "model",
        "is_cache_hit", "cache_similarity", "prompt_tokens", "completion_tokens",
        "cache_hit_tokens", "latency_ms", "ttft_ms", "cloud_tokens", "correct",
        "error", "raw_usage",
    ]
    vals = [int(row[c]) if c == "is_cache_hit" else row[c] for c in cols]
    placeholders = ",".join("?" * len(cols))
    try:
        with _LOCK, session(db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO calls ({','.join(cols)}) VALUES ({placeholders})",
                vals,
            )
    except Exception as e:  # 落库失败不能影响主流程，但要吵
        print(f"[metrics] 落库失败：{e!r}")
    return rec


@dataclass
class TaskRecord:
    """一次用户请求的汇总。三指标里的响应时间与 SLM 使用率都从这行算。"""

    strategy: str
    task_id: str = ""
    run_id: str = "default"
    question: str = ""
    answer: str = ""
    n_subtasks: int = 0
    n_local: int = 0
    n_local_planned: int = 0
    total_ms: float = 0.0
    cloud_tokens: int = 0
    plan_ms: float = 0.0
    exec_ms: float = 0.0
    error: str | None = None
    ts: str = ""

    @property
    def slm_ratio(self) -> float:
        """SLM 使用率 = 走 local 的子任务数 / 总子任务数。direct 模式无子任务，记 0。"""
        return (self.n_local / self.n_subtasks) if self.n_subtasks else 0.0

    def finalize(self) -> "TaskRecord":
        if not self.task_id:
            self.task_id = uuid.uuid4().hex
        if not self.ts:
            self.ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        return self


_TASK_COLS = [
    "task_id", "run_id", "ts", "strategy", "question", "answer", "n_subtasks",
    "n_local", "n_local_planned", "total_ms", "cloud_tokens", "plan_ms",
    "exec_ms", "error",
]


def record_task(task: TaskRecord, db_path: Path | None = None) -> TaskRecord:
    """记录一次完整的用户请求。落库失败不影响主流程，但要吵。"""
    task.finalize()
    row = asdict(task)
    placeholders = ",".join("?" * len(_TASK_COLS))
    try:
        with _LOCK, session(db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO tasks ({','.join(_TASK_COLS)}) "
                f"VALUES ({placeholders})",
                [row[c] for c in _TASK_COLS],
            )
    except Exception as e:  # noqa: BLE001
        print(f"[metrics] 任务落库失败：{e!r}")
    return task


def summary(
    run_id: str | None = None,
    db_path: Path | None = None,
    include_aux: bool = False,
) -> list[sqlite3.Row]:
    """按 strategy 汇总**单次调用**的 token 与耗时。

    默认剔除 warmup / smoke —— 预热调用会把 system prompt 的长度和冷启动耗时
    带进统计，不剔除就等于自己污染自己的指标。要原始数据时传 include_aux=True。
    """
    clauses, args = [], []
    if not include_aux:
        clauses.append("strategy NOT IN ('warmup','smoke')")
    if run_id:
        clauses.append("run_id = ?")
        args.append(run_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
    SELECT
      strategy,
      COUNT(*)                                   AS calls,
      SUM(cloud_tokens)                          AS cloud_tokens_total,
      SUM(prompt_tokens + completion_tokens)     AS all_tokens,
      AVG(latency_ms)                            AS latency_avg_ms,
      SUM(is_cache_hit)                          AS cache_hits
    FROM calls {where}
    GROUP BY strategy ORDER BY strategy
    """
    with session(db_path) as conn:
        return list(conn.execute(sql, args))


def percentile(values: list[float], p: float) -> float:
    """线性插值分位数（numpy 不可用时的兜底，行为与 np.percentile 默认一致）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))


def call_latency_stats(run_id: str | None = None, db_path: Path | None = None) -> dict[str, Any]:
    """**单次调用**的耗时分布，用于排查「哪一步慢」。

    注意这**不是**申请书要求的「完整响应时间」——那个是请求级的，用 task_stats()。
    拿这张表报 P95 会把拆成 5 次调用的请求当成 5 个样本，分位数被压低。
    """
    where = "WHERE error IS NULL AND strategy NOT IN ('warmup','smoke')"
    args: list[Any] = []
    if run_id:
        where += " AND run_id = ?"
        args.append(run_id)
    with session(db_path) as conn:
        rows = list(conn.execute(f"SELECT strategy, latency_ms FROM calls {where}", args))
    out: dict[str, Any] = {}
    for strategy in {r["strategy"] for r in rows}:
        vals = [r["latency_ms"] for r in rows if r["strategy"] == strategy]
        out[strategy] = {
            "n": len(vals),
            "p50_ms": round(percentile(vals, 0.50), 1),
            "p95_ms": round(percentile(vals, 0.95), 1),
        }
    return out


def task_stats(run_id: str | None = None, db_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """请求级三指标，按 strategy 分组——**这才是要写进报告的那张表**。

    - 完整响应时间：中位数 + P95（申请书明确要求这两个）
    - 云端 token 消耗：请求内所有云端调用的 prompt+completion 之和
    - SLM 使用率：走 local 的子任务数 / 总子任务数
    """
    where, args = ("WHERE run_id = ? AND error IS NULL", [run_id]) if run_id else ("WHERE error IS NULL", [])
    with session(db_path) as conn:
        rows = list(conn.execute(f"SELECT * FROM tasks {where}", args))
    out: dict[str, dict[str, Any]] = {}
    for strategy in {r["strategy"] for r in rows}:
        rs = [r for r in rows if r["strategy"] == strategy]
        lat = [r["total_ms"] for r in rs]
        cloud = [r["cloud_tokens"] or 0 for r in rs]
        sub = [r["n_subtasks"] for r in rs]
        # ★ 口径必须是**池化**的 Σn_local / Σn_subtasks，不是「每条请求比值的算术平均」。
        # 两者只在每条请求子任务数相等时才相等。实测：请求 A=2 子任务/2 本地、
        # 请求 B=8 子任务/1 本地 → 平均比值 0.562，池化 0.300，差 26 个百分点。
        # 而 planner 的子任务数本来就随问题在 2~8 之间变，所以这个分叉必然踩到。
        tot_sub = sum(sub)
        tot_local = sum(r["n_local"] for r in rs)
        tot_local_planned = sum(r["n_local_planned"] for r in rs)
        out[strategy] = {
            "n": len(rs),
            "p50_ms": round(percentile(lat, 0.50), 1),
            "p95_ms": round(percentile(lat, 0.95), 1),
            "cloud_tokens_total": int(sum(cloud)),
            "cloud_tokens_median": round(percentile(cloud, 0.50), 1),
            # 两个比例都要报：降级运行时「实际」恒为 100%，只有「分配器判定」
            # 才反映分配策略本身。报告里引用的是后者。
            "slm_ratio_planned": (
                round(tot_local_planned / tot_sub, 3) if tot_sub else None
            ),
            "slm_ratio_actual": round(tot_local / tot_sub, 3) if tot_sub else None,
            "subtasks_median": round(percentile(sub, 0.50), 1) if sub else 0,
        }
    return out


def role_summary(
    run_id: str | None = None,
    db_path: Path | None = None,
    strategy: str | None = None,
    include_aux: bool = False,
) -> list[sqlite3.Row]:
    """按 role 归因 token 与耗时。

    规划开销 = role IN ('decomposer','aggregator') 两行相加。DoT 论文没有
    把这笔开销单独计量，正文也没讨论「拆解本身值不值」——这是我们要报告的点。

    ``strategy`` 用于只统计某一种模式。**算规划开销时必须传 strategy="dot"**：
    run_id 下往往同时跑过 direct 和 dot，而 direct 也有 executor 调用，
    不区分就会把 direct 的执行调用混进 dot 的执行开销里。
    """
    clauses, args = [], []
    # 与 summary() 同口径：默认剔除预热与冒烟。预热固定用 run_id="default"，
    # 不带 --run-id 时会把预热行混进表3，与同屏的表1/表2 口径不一致。
    if not include_aux:
        clauses.append("strategy NOT IN ('warmup','smoke')")
    if run_id:
        clauses.append("run_id = ?")
        args.append(run_id)
    if strategy:
        clauses.append("strategy = ?")
        args.append(strategy)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
    SELECT
      role,
      tier,
      COUNT(*)                                  AS calls,
      SUM(cloud_tokens)                         AS cloud_tokens,
      SUM(prompt_tokens + completion_tokens)    AS all_tokens,
      SUM(latency_ms)                           AS latency_ms_total,
      SUM(CASE WHEN requested_tier <> tier THEN 1 ELSE 0 END) AS fallback_calls,
      SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END)      AS failed_calls
    FROM calls {where}
    GROUP BY role, tier ORDER BY role, tier
    """
    with session(db_path) as conn:
        return list(conn.execute(sql, args))


def plan_overhead(
    run_id: str | None = None,
    db_path: Path | None = None,
    strategy: str | None = "dot",
) -> dict[str, Any]:
    """把规划开销（分解 + 汇聚）单独拎出来，与执行开销分开。

    默认只看 dot ：规划这两个角色本来就只存在于 dot 流水线，而 run_id 里通常
    还混着 direct 的调用；不锁策略的话「执行开销」会混入 direct 那一发，
    数字看着合理、其实口径已经错了。
    """
    rows = role_summary(run_id, db_path, strategy=strategy)
    plan = [r for r in rows if r["role"] in ("decomposer", "aggregator")]
    execu = [r for r in rows if r["role"] == "executor"]
    return {
        "plan_cloud_tokens": sum((r["cloud_tokens"] or 0) for r in plan),
        "plan_calls": sum(r["calls"] for r in plan),
        "plan_ms_total": round(sum((r["latency_ms_total"] or 0) for r in plan), 1),
        "exec_cloud_tokens": sum((r["cloud_tokens"] or 0) for r in execu),
        "exec_calls": sum(r["calls"] for r in execu),
        "exec_ms_total": round(sum((r["latency_ms_total"] or 0) for r in execu), 1),
    }
