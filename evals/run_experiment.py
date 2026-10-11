"""对照实验驱动：多条臂跑同一批题。

四条臂（申请书要求「先实现纯本地、纯云端及规则协同三种模式」，dot 是主方案）：

| 臂 | 档位来源 | 用途 |
|---|---|---|
| `local`  | 强制本地 | 协同收益的下界 |
| `cloud`  | 强制云端 | **申请书指定的对照基准** |
| `direct` | 规则整请求路由 | 「拆不拆」的对照（注意是规则路由器，非强基线） |
| `dot`    | 闸门 + 子任务级分配 | 主方案 |

为什么不直接循环调 CLI：CLI 每次启动都要预热一次本地模型（实测冷加载 30s+），
四条臂 × 17 题 × 3 轮会白白烧掉半小时。这里直接调 cli 的公开函数 ——
**和 CLI 走的是同一条代码路径**（落库、汇总、口径全都在里面），
只是省掉了进程启动和重复预热。

用法：
    python -m evals.run_experiment RUN_ID [重复轮数] [题数上限] [臂,逗号分隔]
例：
    python -m evals.run_experiment exp3 3            # 全部四条臂
    python -m evals.run_experiment exp3 3 17 dot,cloud
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src import cli
from src.llm_client import warmup

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.txt"

ALL_MODES = ("local", "cloud", "direct", "dot")


def load_questions(limit: int | None = None) -> list[str]:
    """读题库。跳过注释行与空行。"""
    qs = []
    for line in QUESTIONS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 去掉行尾的 [预期] 注释
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            qs.append(line)
    return qs[:limit] if limit else qs


def _run_one(question: str, mode: str, task_id: str, run_id: str) -> str:
    """执行一条臂，返回用于进度显示的执行档位串。"""
    if mode == "dot":
        _, results = cli.answer_dot(question, task_id, run_id)
        return "+".join(sorted({r.tier for r in results})) or "闸门拦下"
    if mode in ("local", "cloud"):
        _, meta = cli.answer_pure(question, mode, task_id, run_id)
        return str(meta.get("tier"))
    _, meta = cli.answer_direct(question, task_id, run_id)
    return str(meta.get("tier"))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    run_id = argv[1]
    reps = int(argv[2]) if len(argv) > 2 else 1
    limit = int(argv[3]) if len(argv) > 3 else None
    modes = tuple(argv[4].split(",")) if len(argv) > 4 else ALL_MODES

    unknown = [m for m in modes if m not in ALL_MODES]
    if unknown:
        print(f"未知的臂：{unknown}；可选 {ALL_MODES}")
        return 2

    questions = load_questions(limit)
    print(
        f"题库 {len(questions)} 题 × {len(modes)} 臂 {modes} × {reps} 轮 "
        f"= {len(questions) * len(modes) * reps} 次请求"
    )
    print(f"run_id = {run_id}\n", flush=True)

    # 预热一次即可：本地模型在 keep_alive 窗口内会一直驻留。
    dt = warmup("local")
    print(f"预热完成 {dt:.1f}s\n", flush=True)

    t_start = time.perf_counter()
    for rep in range(1, reps + 1):
        for i, q in enumerate(questions, 1):
            for mode in modes:
                t0 = time.perf_counter()
                task_id = cli._new_task_id()
                try:
                    t = _run_one(q, mode, task_id, run_id)
                except Exception as e:  # noqa: BLE001 - 单条失败不该中断整批
                    t = f"ERR {type(e).__name__}"
                print(
                    f"[{rep}/{reps}] {i:2d}/{len(questions)} {mode:6s} "
                    f"{time.perf_counter() - t0:6.1f}s tier={t:14s} {q[:32]}",
                    flush=True,
                )

    print(f"\n完成，总用时 {time.perf_counter() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
