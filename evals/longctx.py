"""长上下文对照实验：在「云端 token 真正变大」的场景下重跑四臂。

动机（来自 `docs/实验报告_v3_四臂对照.md`）：在短问答题集上纯云端只要
128 token/请求，协同要省的那笔钱天花板极低。所以换到长上下文场景——
input 从几十 token 涨到数千 token，压缩/缓存/下沉才可能拉开差距。

数据：**LongBench**（清华 THUDM，MIT）。选其中两个中文子集：

| 子集 | 任务 | context 均长 |
|---|---|---|
| `dureader` | 多文档问答 | 15,641 字 |
| `multifieldqa_zh` | 单文档问答 | 6,664 字 |

原始数据不在仓库里（114MB），需先下载到 `data/longbench/`（已 gitignore）：
    https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data.zip

选样规则固定在 `SELECT`（每个子集取前 N 条且 context ≤ MAX_CTX_CHARS），
并把选中项落成 `evals/longctx_items.jsonl` 入库——**选样必须可复现**，
不能每次跑都换题。

用法：
    python -m evals.longctx build            # 从 data/ 选样，写 items.jsonl
    python -m evals.longctx run  RUN_ID [轮数]
    python -m evals.longctx score RUN_ID
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "longbench" / "data"
ITEMS_FILE = Path(__file__).resolve().parent / "longctx_items.jsonl"

# 每个子集取几条。取多了没意义：LLM 调用有随机性，但长上下文的差异是结构性的，
# 8 条足以看出方向；而 dot 在长上下文上每次要 k 倍地重发 context，跑太多会烧钱。
SELECT = {"dureader": 4, "multifieldqa_zh": 4}

# context 上限（字符）。本机本地模型 num_ctx=16384，中文约 1 字 ≈ 0.6~1 token，
# 超过这个长度的样本本地必然超窗——那属于「本地直接不可用」，不是我们要测的
# 「协同与纯云端谁更划算」，所以先排除。
MAX_CTX_CHARS = 14000

# 交给模型的 prompt 模板。要求简短作答，避免长上下文场景下输出侧把变量搅浑。
PROMPT_TEMPLATE = """【背景材料】
{context}

【问题】
{question}

请根据背景材料回答问题，只给出答案本身，不要解释，不要复述材料。"""

ARMS = ("local", "cloud", "direct", "dot")


# --------------------------------------------------------------------------
# 选样
# --------------------------------------------------------------------------

def _load_raw(subset: str) -> list[dict]:
    path = RAW_DIR / f"{subset}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"找不到 {path}。请先下载 LongBench 数据包：\n"
            f"  mkdir -p {RAW_DIR.parent}\n"
            f"  curl -sSL --ssl-no-revoke -o {RAW_DIR.parent}/data.zip \\\n"
            f"    https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data.zip\n"
            f"  然后解出 data/dureader.jsonl 与 data/multifieldqa_zh.jsonl"
        )
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build() -> int:
    """从原始数据里选样，写 items.jsonl。确定性：同样输入必得同样输出。"""
    items: list[dict] = []
    for subset, n in SELECT.items():
        raw = _load_raw(subset)
        picked = 0
        for i, r in enumerate(raw):
            ctx = r.get("context", "")
            if not ctx or len(ctx) > MAX_CTX_CHARS:
                continue
            items.append({
                "id": f"{subset}-{i}",
                "source": subset,
                "ctx_chars": len(ctx),
                # LongBench 的问题字段叫 input，不是 question
                "question": r["input"].strip(),
                "context": ctx,
                "answers": r["answers"],
            })
            picked += 1
            if picked >= n:
                break
        if picked < n:
            print(f"[warn] {subset} 只选到 {picked} 条（要求 {n}）", file=sys.stderr)

    ITEMS_FILE.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n",
        encoding="utf-8",
    )
    print(f"写入 {ITEMS_FILE}：{len(items)} 条")
    for x in items:
        print(f"  {x['id']:20s} ctx={x['ctx_chars']:6d}字  {x['question'][:34]}")
    return 0


def load_items() -> list[dict]:
    if not ITEMS_FILE.exists():
        raise SystemExit(f"找不到 {ITEMS_FILE}，先跑 `python -m evals.longctx build`")
    return [json.loads(l) for l in ITEMS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def to_prompt(item: dict) -> str:
    return PROMPT_TEMPLATE.format(context=item["context"], question=item["question"])


# --------------------------------------------------------------------------
# 跑
# --------------------------------------------------------------------------

def run(run_id: str, reps: int = 1) -> int:
    """四臂跑同一批长上下文题。直接调 cli 的公开函数，与 CLI 同一条代码路径。"""
    from src import cli
    from src.llm_client import warmup

    items = load_items()
    print(f"{len(items)} 条 × {len(ARMS)} 臂 × {reps} 轮 = {len(items)*len(ARMS)*reps} 次请求")
    print(f"run_id = {run_id}\n", flush=True)
    print(f"预热完成 {warmup('local'):.1f}s\n", flush=True)

    for rep in range(1, reps + 1):
        for i, item in enumerate(items, 1):
            prompt = to_prompt(item)
            for arm in ARMS:
                task_id = cli._new_task_id()
                try:
                    if arm == "dot":
                        _, results = cli.answer_dot(prompt, task_id, run_id)
                        t = "+".join(sorted({r.tier for r in results})) or "闸门拦下"
                    elif arm in ("local", "cloud"):
                        _, meta = cli.answer_pure(prompt, arm, task_id, run_id)
                        t = str(meta.get("tier"))
                    else:
                        _, meta = cli.answer_direct(prompt, task_id, run_id)
                        t = str(meta.get("tier"))
                except Exception as e:  # noqa: BLE001
                    t = f"ERR {type(e).__name__}"
                print(f"[{rep}/{reps}] {i:2d}/{len(items)} {arm:6s} tier={t:12s} {item['id']}",
                      flush=True)
    return 0


# --------------------------------------------------------------------------
# 评分
# --------------------------------------------------------------------------

_PUNCT = re.compile(r"[\s，。、；：？！“”‘’（）()\[\]【】,.;:?!/\\\"'`~@#$%^&*_\-+=<>|]+")


def _norm(s: str) -> str:
    return _PUNCT.sub("", (s or "").lower())


def contains(pred: str, golds: list[str]) -> bool:
    """命中判定：任一标准答案（去标点后）出现在预测里。

    这是**代理指标**，不是 LongBench 官方指标（dureader 用 Rouge-L、
    multifieldqa_zh 用 F1）。选它是因为两个子集的答案都是短的抽取式片段，
    包含判定足以区分「答对/答错」；报告里按它横向比较四臂，
    **不与论文的公开数字对比**。
    """
    p = _norm(pred)
    return any(_norm(g) and _norm(g) in p for g in golds)


def char_f1(pred: str, gold: str) -> float:
    """字符级 F1（中文 SQuAD 式）。作为 contains 的补充，看部分正确。"""
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return 0.0
    from collections import Counter

    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not same:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score(run_id: str) -> int:
    from src import metrics

    items = {x["id"]: x for x in load_items()}
    with metrics.session() as conn:
        rows = list(conn.execute(
            "SELECT strategy, question, answer FROM tasks WHERE run_id = ? AND error IS NULL",
            (run_id,),
        ))

    # 题目在库里存的是整段 prompt，靠 context 前缀反查是哪个 item
    def find_item(question: str) -> dict | None:
        for it in items.values():
            if it["context"][:200] in question:
                return it
        return None

    tally: dict[str, dict] = {a: {"hit": 0, "n": 0, "f1": 0.0} for a in ARMS}
    for r in rows:
        it = find_item(r["question"])
        if it is None:
            continue
        t = tally.setdefault(r["strategy"], {"hit": 0, "n": 0, "f1": 0.0})
        t["n"] += 1
        t["hit"] += int(contains(r["answer"], it["answers"]))
        t["f1"] += max((char_f1(r["answer"], g) for g in it["answers"]), default=0.0)

    print(f"run_id = {run_id}    （命中=包含任一标准答案；F1=字符级，取最大）\n")
    print(f"{'arm':8s} {'命中':>6s} {'样本':>6s} {'命中率':>8s} {'平均F1':>8s}")
    print("-" * 42)
    for a in ARMS:
        t = tally.get(a)
        if not t or not t["n"]:
            continue
        print(f"{a:8s} {t['hit']:6d} {t['n']:6d} {t['hit']/t['n']*100:7.1f}% "
              f"{t['f1']/t['n']:8.3f}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "build":
        return build()
    if cmd == "run":
        if len(argv) < 3:
            print("用法：python -m evals.longctx run RUN_ID [轮数]")
            return 2
        return run(argv[2], int(argv[3]) if len(argv) > 3 else 1)
    if cmd == "score":
        if len(argv) < 3:
            print("用法：python -m evals.longctx score RUN_ID")
            return 2
        return score(argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
