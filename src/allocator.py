"""请求路由：把「这条（子）任务该交给本地 SLM 还是云端大模型」判出来。

**纯规则，零 LLM 调用**——这是设计而不是偷懒。

为什么不做学习式路由：DoT 论文里的 adapter（P3）需要先用一批标注好的
「任务 → 档位」数据离线训练，论文自己报告不训练时 P3 的准确率只有 15%。
本项目没有这批训练数据，硬套一个没训过的 router 只会得到随机档位，
把「端云协同」的对照实验做成一团噪声。所以 v4 退回到**可解释的阈值规则**，
并在报告里如实写明「这是规则路由，不是学习出来的路由器」——否则它会被
误当成 DataShunt 那样的强基线，结论就站不住了。

规则之外还有一层意义：规则是**确定性**的。同一个问题跑多少次都得到同一
档位，实验的可复现性才有保证（LLM 路由会引入随机性，seed 也压不住）。

规则表（**优先级严格按顺序**，先命中先返回，见 §2 allocator）：

| 条件 | 判定 | 理由 |
|---|---|---|
| 长度 > LONG_TEXT_CHARS | cloud | 长文本(>60字) |
| 命中 OPEN_ENDED_WORDS | cloud | 开放性问题(<词>) |
| 纯算术 | local | 纯算术 |
| 其余 | local | 默认本地 |

阈值与词表全部提成模块级常量，方便以后做阈值扫描（换 60 字 → 40/80 字）。
"""

from __future__ import annotations

# ── 模块级常量：调参/扫描只改这里，不要散到函数体里当魔法数字 ──────────────

# 超过这个字符数就走云端。60 是 v1 的经验阈值：本地 3B 模型在长输入上
# 更容易丢约束（复述题干、漏步），短问题则本地足够且省云端 token。
LONG_TEXT_CHARS = 60

# 开放（主观/多步推理）问题词表。命中即判 cloud——这类问题 SLM 的答案质量
# 与云端差距最大，也正是「协同」收益的来源，所以宁可多花云端 token。
# 注意这是**有序**元组：多词同时命中时，按声明顺序取第一个，保证结果确定。
OPEN_ENDED_WORDS = (
    "说明",
    "解释",
    "分析",
    "比较",
    "论述",
    "评价",
    "为什么",
    "原因",
    "设计",
    "证明",
    "推理",
    "权衡",
    "优缺点",
)

# 「纯算术」的白名单字符。判定用的是**白名单**而不是黑名单：
# 只有整句都由这些字符组成才叫纯算术，任何别的字（尤其是中文修饰词，
# 如「计算」「请问」「帮我」）都会把它踢出算术类，落到默认本地。
#
# 为什么白名单里要放「的多少等于」这几个中文字：中文算术题几乎必然带
# 「37*48 等于多少？」这种问法。若不放行，「等于」成了非法字符，
# 整句就不算纯算术了——虽然它本来就该判 local，但理由会从「纯算术」
# 退化成「默认本地」，路由理由失真，事后没法从指标里看出规则是否按预期工作。
ARITHMETIC_CN_UNITS = frozenset("的多少等于")

# 允许的算术符号与问号。问号放行是因为「37*48 等于多少？」整体仍是算术，
# 不能因为一个句尾问号就否定它。
ARITHMETIC_SYMBOLS = frozenset("+-*/()=？?")

# 显式列出半角/全角数字，比 str.isdigit() 可控（isdigit 会把上标 ² 也算数字）。
ARITHMETIC_DIGITS = frozenset("0123456789０１２３４５６７８９")

# explain() 里给 CLI 展示的文本预览长度
PREVIEW_CHARS = 32

# 档位字面量。全系统只有 "local" / "cloud" 两种（降级由 llm_client 处理，
# allocator 只管「请求哪一档」，不关心它最后会不会被降级）。
TIER_LOCAL = "local"
TIER_CLOUD = "cloud"


def _is_pure_arithmetic(text: str) -> bool:
    """整句是否只由数字与算术符号组成（且至少有一个数字）。

    判定为白名单式：任何一个不在允许集合里的字符都直接判否，因此带中文
    修饰词的句子（「请计算…」「帮我算一下…」）不会被误判为纯算术。
    """
    has_digit = False
    for ch in text:
        if ch in ARITHMETIC_DIGITS:
            has_digit = True
            continue
        if ch in ARITHMETIC_SYMBOLS or ch in ARITHMETIC_CN_UNITS:
            continue
        if ch.isspace():  # 空白：含全角空格与各类 unicode 空白
            continue
        return False
    return has_digit


def _matched_open_word(text: str) -> str | None:
    """返回命中的第一个开放词（按 OPEN_ENDED_WORDS 声明顺序），没命中返回 None。

    顺序敏感不是随意的：用它保证同一条文本每次得到同一个理由文案。
    """
    for word in OPEN_ENDED_WORDS:
        if word in text:
            return word
    return None


def _classify(text: str) -> tuple[str, str]:
    """核心判定：返回 (tier, 理由文案)。所有对外函数的判定都收敛到这里。

    优先级严格按模块 docstring 里的规则表，先命中先返回——
    例如一句又长又开放的文本，理由应记「长文本」而不是「开放性问题」，
    因为长度是先判的。理由文案直接进报告，所以顺序不能乱。
    """
    if len(text) > LONG_TEXT_CHARS:
        return TIER_CLOUD, f"长文本(>{LONG_TEXT_CHARS}字)"

    word = _matched_open_word(text)
    if word is not None:
        return TIER_CLOUD, f"开放性问题({word})"

    if _is_pure_arithmetic(text):
        return TIER_LOCAL, "纯算术"

    return TIER_LOCAL, "默认本地"


def _preview(text: str) -> str:
    """单行化并截断，供 explain 展示——避免多行子任务把 CLI 表格撑乱。"""
    flat = " ".join(text.split())
    if len(flat) <= PREVIEW_CHARS:
        return flat
    return flat[:PREVIEW_CHARS] + "…"


def allocate_one(text: str) -> str:
    """判定**整条请求**的档位，返回 "local" | "cloud"。

    这就是 direct 模式要用的整请求路由（一次性把整个问题交给某一档），
    与 dot 模式按子任务逐条 allocate 相对应。两种模式共用同一套规则，
    对照才是公平的——差别只在「拆不拆」，不在「规则不同」。
    """
    return _classify(text)[0]


def allocate(subtasks: list[str]) -> list[str]:
    """逐条判定子任务档位，返回与 subtasks 等长的 "local"/"cloud" 列表。"""
    return [_classify(sub)[0] for sub in subtasks]


def explain(subtasks: list[str], tiers: list[str]) -> list[str]:
    """把每条子任务的判定理由摊开给 CLI 展示，返回与 subtasks 等长的列表。

    参数里的 tiers 用**给定值**展示（而不是重新算一遍）——它是即将真正执行的
    档位，以此为准；同时拿规则重判一次做交叉核对，两者不一致时显式标注。
    这样一旦上游传了错档位（比如手改过的 tiers），能立刻在界面上看到，
    而不是等实验跑完对不上账才发现。
    """
    out: list[str] = []
    for i, sub in enumerate(subtasks):
        tier = tiers[i] if i < len(tiers) else None
        rule_tier, reason = _classify(sub)

        if tier is None:
            shown = rule_tier
            mark = "（未给定档位，采用规则判定）"
        else:
            shown = tier
            mark = "" if tier == rule_tier else f"（⚠ 给定 {tier} 与规则 {rule_tier} 不符）"

        out.append(f"{i + 1}. {shown}｜{reason}｜{_preview(sub)}{mark}")
    return out


if __name__ == "__main__":
    # 自测：只依赖本文件，不 import 其它未完成模块，方便单独验证规则表。
    samples = [
        "37*48 等于多少？",
        "请计算 37*48",
        "一个班 45 人，其中 3/5 是女生，女生比男生多几人？",
        "说明端云协同的基本原理",
        "比较本地模型和云端模型在延迟上的差异" + "，" * 40,
    ]
    for s in samples:
        print(f"{allocate_one(s):>5}  {s[:24]}")
    print()
    tiers = allocate(samples)
    for line in explain(samples, tiers):
        print(line)
