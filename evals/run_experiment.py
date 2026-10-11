"""对照实验驱动：direct（整请求路由）vs dot（子任务级分配）跑同一批题。

为什么不直接循环调 CLI：CLI 每次启动都要预热一次本地模型（实测冷加载 30s+），
18 次调用会白白烧掉十分钟。这里直接调 cli.answer_direct / cli.answer_dot ——
**和 CLI 走的是同一条代码路径**（落库、汇总、口径全都在里面），只是省掉了
进程启动和重复预热。

用法：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m evals.run_experiment RUN_ID [重复轮数] [题数上限]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from src import cli
from src.llm_client import warmup

QUESTIONS_FILE = Path(__file__).resolve().parent / "questions.txt"


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


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    run_id = argv[1]
    reps = int(argv[2]) if len(argv) > 2 else 1
    limit = int(argv[3]) if len(argv) > 3 else None

    questions = load_questions(limit)
    print(f"题库 {len(questions)} 题 × 2 模式 × {reps} 轮 = {len(questions) * 2 * reps} 次请求")
    print(f"run_id = {run_id}\n", flush=True)

    # 预热一次即可：本地模型在 keep_alive 窗口内会一直驻留。
    dt = warmup("local")
    print(f"预热完成 {dt:.1f}s\n", flush=True)

    t_start = time.perf_counter()
    for rep in range(1, reps + 1):
        for i, q in enumerate(questions, 1):
            for mode in ("direct", "dot"):
                t0 = time.perf_counter()
                task_id = cli._new_task_id()
                if mode == "direct":
                    _, meta = cli.answer_direct(q, task_id, run_id)
                    t = meta.get("tier")
                else:
                    _, results = cli.answer_dot(q, task_id, run_id)
                    t = "+".join(sorted({r.tier for r in results})) or "-"
                print(
                    f"[{rep}/{reps}] {i:2d}/{len(questions)} {mode:6s} "
                    f"{time.perf_counter() - t0:6.1f}s tier={t:11s} {q[:34]}",
                    flush=True,
                )

    print(f"\n完成，总用时 {time.perf_counter() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
