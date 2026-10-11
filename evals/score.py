"""对实验答案评分，产出「性能」这一项指标。

题库里的 [预期] 是人工直觉标注、不是标准答案（questions.txt 里已声明），
不能拿来评分。所以标准答案单独维护在本文件里。

评分范围：9 题中只有 7 题有确定数值答案，2 道开放题
（端云协同优势/风险、量化 vs 蒸馏）无法自动判分，**排除在准确率之外**，
在报告里如实标注样本量。

用法：
    PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -m evals.score RUN_ID
"""

from __future__ import annotations

import re
import sys

from src import metrics

# 题面关键词 → 标准答案。用关键词匹配而不是全等，避免题库改标点时对不上。
GOLD: dict[str, float] = {
    "37 乘以 48": 1776,
    "吃掉 7 个": 17,
    "156 除以 12": 13,
    "3/5 是女生": 9,
    "连续跑 4 周": 60,
    "打八折": 236,
    "18 的平方": 224,
}

# 无标准答案的开放题（仅用于计数与说明）
OPEN: tuple[str, ...] = ("端云协同架构", "量化通常比蒸馏")


def gold_of(question: str) -> float | None:
    for key, val in GOLD.items():
        if key in question:
            return val
    return None


def is_open(question: str) -> bool:
    return any(k in question for k in OPEN)


def hit(answer: str, gold: float) -> bool:
    """答案里是否出现了标准答案这个数（按数字边界匹配，避免 17 命中 170）。"""
    if gold == int(gold):
        pat = rf"(?<![\d.]){int(gold)}(?![\d.])"
    else:
        pat = rf"(?<![\d.]){re.escape(str(gold))}(?![\d.])"
    return re.search(pat, answer or "") is not None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    run_id = argv[1]

    with metrics.session() as conn:
        rows = list(
            conn.execute(
                "SELECT strategy, question, answer FROM tasks "
                "WHERE run_id = ? AND error IS NULL ORDER BY question, strategy",
                (run_id,),
            )
        )

    # question → {strategy: [answers]}（多轮重复时收集全部）
    by_q: dict[str, dict[str, list[str]]] = {}
    for r in rows:
        by_q.setdefault(r["question"], {}).setdefault(r["strategy"], []).append(
            r["answer"] or ""
        )

    strategies = sorted({r["strategy"] for r in rows})
    tally = {s: {"ok": 0, "n": 0} for s in strategies}
    detail: list[tuple[str, str, bool | None]] = []

    for q, per_mode in sorted(by_q.items()):
        g = gold_of(q)
        if g is None:
            for s in strategies:
                if per_mode.get(s):
                    detail.append((s, q, None))
            continue
        for s in strategies:
            for a in per_mode.get(s, []):
                ok = hit(a, g)
                tally[s]["ok"] += int(ok)
                tally[s]["n"] += 1
                detail.append((s, q, ok))

    print(f"run_id = {run_id}")
    print(f"计分题 {len(GOLD)} 道（开放题 {len(OPEN)} 道已排除）\n")
    print(f"{'strategy':10s} {'正确':>4s} {'样本':>4s} {'准确率':>8s}")
    print("-" * 30)
    for s in strategies:
        t = tally[s]
        acc = (t["ok"] / t["n"] * 100) if t["n"] else 0.0
        print(f"{s:10s} {t['ok']:4d} {t['n']:4d} {acc:7.1f}%")

    print("\n逐题：")
    for s, q, ok in detail:
        mark = "—" if ok is None else ("✓" if ok else "✗")
        print(f"  [{s:6s}] {mark} {q[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
